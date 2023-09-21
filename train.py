import copy
import gc
import os
import warnings

import argparse
import collections
import shutil
import sys
import time

import numpy as np
import setproctitle
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datetime import datetime


# import models
from model import resnet, resnet_cifar10, vgg, deit

from utils import dist_utils, experimental_utils, dataloader
from utils.fp16util import *
from utils.logger import TensorboardLogger, FileLogger
from utils.meter import AverageMeter, NetworkMeter, TimeMeter
from utils.model_utils import get_grad_norm, get_param_norm
from utils.data_utils import DataManager_ImageNet, DataManager_CIFAR, DataManager_CINIC10, DataManager_FLOWER102

from ptflops import get_model_complexity_info
from utils.parse import get_parser


cudnn.benchmark = True
args = get_parser().parse_args()
# os.sched_setaffinity( os.getpid(), {i for i in range(49, 97)} )

# Only want master rank logging to tensorboard
is_master = ( not args.distributed) or (dist_utils.env_rank() == 0 )
is_rank0 = args.local_rank == 0
tb = TensorboardLogger( args.logdir, is_master=is_master )
log = FileLogger( args.logdir, is_master=is_master, is_rank0=is_rank0 )


def main():

    os.system( 'shutdown -c' )  # cancel previous shutdown command
    log.console( args )
    tb.log( 'sizes/world', dist_utils.env_world_size() )

    str_process_name = args.model + "-" + args.dataset + "-" + args.optimizer + ":" + str(args.local_rank)
    setproctitle.setproctitle( str_process_name )

    # need to index validation directory before we start counting the time
    # dataloader.sort_ar(args.data + '/val')

    if args.distributed:
        log.console( 'Distributed initializing process group. Rank = %d' % args.local_rank )
        torch.cuda.set_device( args.local_rank )
        dist.init_process_group( backend=args.dist_backend, init_method=args.dist_url,
                                 world_size=dist_utils.env_world_size() )
        assert ( dist_utils.env_world_size() == dist.get_world_size() )
        log.console( "Distributed: success (%d/%d)" % ( args.local_rank, dist.get_world_size() ) )

    sz_img, n_classes = 32, 10
    if args.dataset == 'imagenet':
        sz_img, n_classes = 224, 1000
    if args.dataset == 'cifar100':
        sz_img, n_classes = 32, 100

    log.console( "Loading model" )

    if args.dataset == 'imagenet':
        model = resnet.__dict__[ args.model ]( pretrained=False, num_classes=1000 ).cuda()
    elif args.dataset == 'cifar10':
        if 'resnet' in args.model:
            model = resnet_cifar10.__dict__[ args.model ]( pretrained=False, num_classes=10 ).cuda()
        elif 'deit' in args.model:
            args.patch = 4
            model = deit.deit_tiny_patch16_224( pretrained=True, img_size=32, num_classes=10, patch_size=4, args=args )
    elif args.dataset == 'cifar100':
        if 'resnet' in args.model:
            model = resnet_cifar10.__dict__[args.model]( pretrained=False, num_classes=100 ).cuda()
        elif 'deit' in args.model:
            args.patch = 4
            model = deit.deit_tiny_patch16_224( pretrained=True, img_size=32, num_classes=100, patch_size=4, args=args )

    if args.fp16:
        model = network_to_half( model )
    if args.distributed:
        model = dist_utils.DDP( model, device_ids=[args.local_rank], output_device=args.local_rank )
    else:
        model = torch.nn.DataParallel( model )

    # calculate the parameter size
    flops, params = get_model_complexity_info(
        model, ( 3, sz_img, sz_img ), as_strings=False,
        print_per_layer_stat=False, verbose=False
    )
    if args.local_rank == 0:
        print('{:<30}  {:<8}'.format( 'Computational complexity: ', flops ) )
        print('{:<30}  {:<8}'.format( 'Number of parameters: ', params ) )

    global model_params, master_params
    if args.fp16:
        model_params, master_params = prep_param_lists( model )
    else:
        model_params = master_params = model.parameters()

    # - create parameter blocks if using SLIMBLOCK
    """
    if args.local_rank == 0:
        print( model ); quit()
    """

    from utils.model_utils import get_pblocks, get_pblocks_resnet50, get_pblocks_deit
    pblocks = None
    if args.optimizer == 'SLIMBLOCK':
        if 'resnet18' in args.model:
            pblocks = get_pblocks( model.parameters() )
        elif 'resnet50' in args.model:
            pblocks = get_pblocks_resnet50( master_params )
        elif 'deit' in args.model:
            pblocks = get_pblocks_deit( model.parameters() )

    """Debug block division
    if args.local_rank == 0:
        for key in pblocks.keys():
            print('block', key)
            for p in pblocks[key]:
                print(p.shape)
        quit()
    """

    # define loss function (criterion) and optimizer
    from utils.model_utils import get_optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer, preconditioner, kfac_param_scheduler = get_optimizer( args, model, master_params, pblocks )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict( checkpoint[ 'state_dict' ] )
        args.start_epoch, best_top5 = checkpoint[ 'epoch' ], checkpoint[ 'best_top5' ]
        optimizer.load_state_dict( checkpoint[ 'optimizer' ] )

    # save script so we can reproduce from logs
    shutil.copy2( os.path.realpath(__file__), f'{args.logdir}' )

    log.console( "Creating data loaders (this could take up to 10 minutes if volume needs to be warmed up)" )
    phases = eval( args.phases )
    if args.dataset == 'imagenet':
        dm = DataManager_ImageNet( [ copy.deepcopy(p) for p in phases if 'bs' in p ], args )
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        dm = DataManager_CIFAR( [ copy.deepcopy(p) for p in phases if 'bs' in p ], args )
    else:
        print( '[ERROR] provide a valid dataset' )

    # scheduler = Scheduler(optimizer, [copy.deepcopy(p) for p in phases if 'lr' in p])
    # lr_lambda = lambda epoch: args.lr_decay ** (epoch // args.decay_period)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, args.max_epoch, eta_min=0.0001 )
    # scheduler = torch.optim.lr_scheduler.CyclicLR( optimizer, base_lr=0.01, max_lr=0.1, step_size_up=4000 )

    start_time = datetime.now()  # Loading start to after everything is loaded
    if args.evaluate:
        dm.set_epoch( 0 )
        return validate( dm.val_dl, model, criterion, 0, start_time )

    if args.distributed:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor( torch.tensor( [ 1.0 ] ).float().cuda() )

    log.event("~~epoch\thours\ttop1\ttop5\n")
    dm.set_epoch( 0 )
    _, _, loss_init = validate( dm.val_dl, model, criterion, 0, start_time, istrain=False )
    for epoch in range(0, args.max_epoch):
        dm.set_epoch(epoch)

        # scheduler.step()
        # user defined scheduler
        # scheduler.update_lr( epoch, 0, len(dm.trn_dl) )

        time_epoch_start = time.time()
        loss_i = train( dm.trn_dl, model, criterion, optimizer, preconditioner, scheduler, epoch )
        log.verbose("#############################one epoch time cost: %f" % ( time.time() - time_epoch_start ) )

        top1, top5, _ = validate( dm.val_dl, model, criterion, epoch, start_time, istrain=True )

        # scheduler.update_lr( epoch, 1, len( dm.trn_dl ) )
        scheduler.step()

        if args.optimizer == 'KFAC':
            kfac_param_scheduler.step()

        time_diff = ( datetime.now() - start_time).total_seconds() / 3600.0
        log.event( f'~~{epoch}\t{time_diff:.5f}\t\t{top1:.3f}\t\t{top5:.3f}\n' )

        """save checkpoint
        is_best = top5 > best_top5
        best_top5 = max( top5, best_top5 )
        if args.local_rank == 0:
            if is_best:
                save_checkpoint(epoch, model, best_top5, optimizer, is_best=True, filename='model_best.pth.tar')
            if epoch % 5 == 0:
               save_checkpoint(epoch, model, best_top5, optimizer, filename=args.logdir+f'/epoch{epoch}_checkpoint.tar')
        """


def train( trn_loader, model, criterion, optimizer, preconditioner, scheduler, epoch ):
    timer = TimeMeter()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    start = time.time()

    # switch to train mode
    model.train()
    n_batches = len( trn_loader )
    for i, ( input, target ) in enumerate( trn_loader ):
        input, target = input.cuda(), target.cuda()
        batch_num = i + 1

        timer.batch_start()

        # compute output
        time_start_forward = time.time()
        output = model( input )
        time_end_forward = time.time()
        # for single machine training, this value is around 10ms
        # log.verbose("  forward time cost: %f" % (time_end_forward - time_start_forward))

        loss = criterion(output, target)

        # compute gradient and do SGD step
        if args.fp16:
            model.zero_grad()

            loss = loss * args.loss_scale
            loss.backward()
            model_grads_to_master_grads( model_params, master_params )
            for param in master_params:
                param.grad.data = param.grad.data / args.loss_scale
            if args.optimizer == 'SGD' or args.optimizer == 'KFAC':
                optimizer.step()
            elif args.optimizer == 'SLIM' or args.optimizer == 'SLIMBLOCK':
                optimizer.step( epoch=epoch )
            master_params_to_model_params(model_params, master_params)
            loss = loss / args.loss_scale

            time_end_backward = time.time()
            # for single machine training, this value is 50ms, when training
            # log.verbose("backwards time cost: %f" % (time_end_backward - time_end_forward))
        else:
            optimizer.zero_grad()

            # start to all_reduce with the order of bucket
            loss.backward()
            time_end_backward = time.time()
            # log.verbose("backwards time cost: %f" % (time_end_backward - time_end_forward))
            if args.optimizer == 'SGD' or args.optimizer == 'ADAM':
                optimizer.step()
            elif args.optimizer == 'SLIM' or args.optimizer == 'SLIMBLOCK':
                optimizer.step( epoch=epoch )
            elif args.optimizer == 'KFAC':
                pn_kfac = get_param_norm(model)
                gn_before_kfac = get_grad_norm(model)
                preconditioner.step()
                gn_after_kfac = get_grad_norm(model)
                optimizer.step()

        # lr schedule
        tb.log( "sizes/lr", optimizer.param_groups[ 0 ][ 'lr' ] )
        # scheduler.step( epoch + i/n_batches )
        # scheduler.step()
        # scheduler.update_lr( epoch, i, n_batches )

        # Train batch done. Logging results
        timer.batch_end()
        corr1, corr5 = correct( output.data, target, topk=(1, 5) )
        reduced_loss, batch_total = to_python_float(loss.data), to_python_float(input.size(0))

        # Must keep track of global batch size
        # since not all machines are guaranteed equal batches at the end of an epoch
        if args.distributed:
            metrics = torch.tensor([batch_total, reduced_loss, corr1, corr5]).float().cuda()
            batch_total, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss / dist_utils.env_world_size()
        top1acc = to_python_float( corr1 ) * ( 100.0 / batch_total )
        top5acc = to_python_float( corr5 ) * ( 100.0 / batch_total )

        losses.update( reduced_loss, batch_total )
        top1.update( top1acc, batch_total )
        top5.update( top5acc, batch_total )

        should_print = (batch_num % args.print_freq == 0) or (batch_num == len(trn_loader))
        if args.local_rank == 0:
            if should_print:
                tb.log_trn_loss(losses.avg, top1.avg, top5.avg)

                output = ( f'Epoch: [{epoch}][{batch_num}/{len(trn_loader)}]\t'
                           f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                           f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                           f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                           f'Data {timer.data_time.val:.3f} ({timer.data_time.avg:.3f})\t' )
                log.verbose(output)

                if args.optimizer == 'KFAC':
                    tb.log_kfac( gn_before_kfac, gn_after_kfac, pn_kfac )
                if args.optimizer == 'SGD':
                    tb.log_sgd( optimizer.gn, optimizer.pn )
                if args.optimizer == 'SLIM' and optimizer.update_dg_dp and optimizer.start_slim:
                    # log.verbose( output )
                    tb.log_lbfgs( optimizer.rho_list[-1], optimizer.h0,
                                  optimizer.tao_before, optimizer.tao_after,
                                  optimizer.gn_before, optimizer.gn_after, optimizer.pn )
                if args.optimizer == 'SLIMBLOCK' and optimizer.start_slim:
                    tb.log_blockslim( optimizer )
        tb.update_step_count( batch_total )

    return losses.avg


def validate(val_loader, model, criterion, epoch, start_time, istrain = True):
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    eval_start_time = time.time()

    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        batch_num = i + 1
        timer.batch_start()
        if args.distributed:
            top1acc, top5acc, loss, batch_total = distributed_predict(input, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target).data
            batch_total = input.size(0)
            top1acc, top5acc = accuracy(output.data, target, topk=(1, 5))

        # Eval batch done. Logging results
        timer.batch_end()
        losses.update(to_python_float(loss), to_python_float(batch_total))
        top1.update(to_python_float(top1acc), to_python_float(batch_total))
        top5.update(to_python_float(top5acc), to_python_float(batch_total))
        should_print = (batch_num % args.print_freq == 0) or (batch_num == len(val_loader))
        if args.local_rank == 0 and should_print:
            output = (f'Test:  [{epoch}][{batch_num}/{len(val_loader)}]\t'
                      f'Time {timer.batch_time.val:.3f} ({timer.batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            log.verbose(output)

    if istrain:
        tb.log_eval(top1.avg, top5.avg, time.time() - eval_start_time)
        tb.log('epoch', epoch)

    return top1.avg, top5.avg, losses.avg


def distributed_predict(input, target, model, criterion):
    # Allows distributed prediction on uneven batches. Test set isn't always large enough for every GPU to get a batch
    batch_size = input.size(0)
    output = loss = corr1 = corr5 = valid_batches = 0

    if batch_size:
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target).data
        # measure accuracy and record loss
        valid_batches = 1
        corr1, corr5 = correct(output.data, target, topk=(1, 5))

    metrics = torch.tensor([batch_size, valid_batches, loss, corr1, corr5]).float().cuda()
    batch_total, valid_batches, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
    reduced_loss = reduced_loss / valid_batches

    top1 = corr1 * (100.0 / batch_total)
    top5 = corr5 * (100.0 / batch_total)
    return top1, top5, reduced_loss, batch_total

# ### Learning rate scheduler
def create_lr_schedule(workers, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        decay_schedule.sort(reverse=True)
        for e in decay_schedule:
            if epoch >= e:
                lr_adj *= alpha
        return lr_adj
    return lr_schedule

class Scheduler():
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        if len(phase['lr']) == 2:
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['ep']
        if 'epoch_step' in phase: batch_curr = 0  # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        # change lr for every iteration
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr
        step_size = (lr_end - lr_start) / step_tot

        # change lr for every epoch
        # step_size = (lr_end - lr_start) / epoch_tot
        return lr_start + step_curr * step_size

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')

    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1: return phase['lr'][0]  # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        tb.log("sizes/lr", lr)


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if isinstance(t, (float, int)): return t
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def save_checkpoint(epoch, model, best_top5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1, 'state_dict': model.state_dict(),
        'best_top5': best_top5, 'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    if is_best: shutil.copyfile(filename, f'{args.logdir}/{filename}')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]


def correct(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res


def listify(p=None, q=None):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    n = q if type(q) == int else 1 if q is None else len(q)
    if len(p) == 1: p = p * n
    return p


if __name__ == '__main__':
    main()

    tb.close()
