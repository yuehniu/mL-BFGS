import time
import torch
from tensorboardX import SummaryWriter


class TensorboardLogger:
    def __init__(self, output_dir, is_master=False):
        self.output_dir = output_dir
        self.current_step = 0
        if is_master:
            self.writer = SummaryWriter(self.output_dir)
        else:
            self.writer = NoOp()
        #self.log('first', time.time())

    def log(self, tag, val):
        """Log value to tensorboard (relies on global_example_count being set properly)"""
        if not self.writer: return
        self.writer.add_scalar(tag, val, self.current_step)

    def update_step_count(self, batch_total):
        self.current_step += batch_total

    def close(self):
        self.writer.export_scalars_to_json(self.output_dir + '/scalars.json')
        self.writer.close()

    # Convenience logging methods
    def log_size(self, bs=None, sz=None):
        if bs: self.log('sizes/batch', bs)
        if sz: self.log('sizes/image', sz)

    def log_eval(self, top1, top5, time):
        self.log('losses/test_1', 100.0-top1)
        # self.log('losses/test_5', 100.0-top5)
        self.log('times/eval_sec', time)

    def log_trn_loss(self, loss, top1, top5):
        self.log("losses/xent", loss)  # cross_entropy
        self.log("losses/train_1", 100.0-top1)  # precision@1
        # self.log("losses/train_5", 100.0-top5)  # precision@5

    def log_memory(self):
        if not self.writer: return
        self.log("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9)
        self.log("memory/max_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
        self.log("memory/cached_gb", torch.cuda.memory_reserved() / 1e9)
        self.log("memory/max_cached_gb", torch.cuda.max_memory_reserved() / 1e9)

    def log_trn_times(self, batch_time, data_time, batch_size):
        if not self.writer: return
        self.log("times/step", 1000 * batch_time)
        self.log("times/data", 1000 * data_time)
        images_per_sec = batch_size / batch_time
        self.log("times/1gpu_images_per_sec", images_per_sec)
        self.log("times/8gpu_images_per_sec", 8 * images_per_sec)

    def log_lbfgs(self, rho, h0, tao_before, tao_after, gn_before, gn_after, pn):
        """
        observer SLIMQN optimization
        please refer to the paper to check each metric
        @return:
        """
        self.log("lbfgs/rho", rho)
        self.log("lbfgs/h0", h0)
        self.log("lbfgs/tao_before", tao_before)
        self.log("lbfgs/tao_after", tao_after)
        self.log("lbfgs/g_norm_before", gn_before)
        self.log("lbfgs/g_norm_after", gn_after)
        self.log("lbfgs/p_norm", pn)

    def log_blockslim( self, opt ):
        """
        observe BlockSlimQN optimization
        please refer to the paper to check each metric
        @return:
        """
        self.log( "blockslim/p_norm", opt.pn )
        for bk in opt.rho_list.keys():
            self.log( "blockslim/rho/"+bk, opt.rho_list[ bk ][ -1 ] )
            self.log( "blockslim/tao_before/" + bk, opt.tao_before[ bk ] )
            self.log( "blockslim/tao_after/" + bk, opt.tao_after[ bk ] )
            self.log( "blockslim/gn_before/" + bk, opt.gn_before[ bk ] )
            self.log( "blockslim/gn_after/" + bk, opt.gn_after[ bk ] )

            if opt.has_hist_g[ bk ]:
                self.log( "blockslim/avg_g/" + bk, opt.avg_gn[ bk ] )
                self.log( "blockslim/avg_p/" + bk, opt.avg_pn[ bk ] )

    def log_kfac(self, gn_before, gn_after, pn):
        self.log("kfac/g_norm_before", gn_before)
        self.log("kfac/g_norm_after", gn_after)
        self.log("kfac/p_norm", pn)

    def log_sgd(self, gn, pn):
        self.log("sgd/g_norm", gn)
        self.log("sgd/p_norm", pn)


import logging


class FileLogger:
    def __init__(self, output_dir, is_master=False, is_rank0=False):
        self.output_dir = output_dir

        # Log to console if rank 0, Log to console and file if master
        if not is_rank0:
            self.logger = NoOp()
        else:
            self.logger = self.get_logger(output_dir, log_to_file=is_master)

    def get_logger(self, output_dir, log_to_file=True):
        logger = logging.getLogger('imagenet_training')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if log_to_file:
            vlog = logging.FileHandler(output_dir + '/verbose.log')
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            logger.addHandler(vlog)

            eventlog = logging.FileHandler(output_dir + '/event.log')
            eventlog.setLevel(logging.WARN)
            eventlog.setFormatter(formatter)
            logger.addHandler(eventlog)

            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir + '/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass

        return no_op
