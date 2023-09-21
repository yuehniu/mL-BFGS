import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gc
from . import dataloader

def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=2, root='../data'):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)


    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader

class DataManager_ImageNet():
    def __init__(self, phases, args):
        self.args = args
        self.phases = self.preload_phase_data(phases)

    def set_epoch(self, epoch):
        cur_phase = self.get_phase(epoch)
        if cur_phase:
            self.set_data(cur_phase)
        if hasattr(self.trn_smp, 'set_epoch'):
            self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'):
            self.val_smp.set_epoch(epoch)

    def get_phase(self, epoch):
        return next((p for p in self.phases if p['ep'] == epoch), None)

    def set_data(self, phase):
        """Initializes data loader."""
        if phase.get('keep_dl', False):
            self.trn_dl.update_batch_size(phase['bs'])
            return

        self.trn_dl, self.val_dl, self.trn_smp, self.val_smp = phase['data']
        self.phases.remove(phase)

        # clear memory before we begin training
        gc.collect()

    def preload_phase_data(self, phases):
        for phase in phases:
            if not phase.get('keep_dl', False):
                self.expand_directories(phase)
                phase['data'] = self.preload_data(**phase)
        return phases

    def expand_directories(self, phase):
        trndir = phase.get('trndir', '')
        valdir = phase.get('valdir', trndir)
        phase['trndir'] = self.args.data + trndir + '/train'
        phase['valdir'] = self.args.data + valdir + '/val'

    def preload_data(self, ep, sz, bs, trndir, valdir, **kwargs):  # dummy ep var to prevent error
        if 'lr' in kwargs: del kwargs['lr']  # in case we mix schedule and data phases
        """Pre-initializes data-loaders. Use set_data to start using it."""
        if sz == 128:
            val_bs = max(bs, 512)
        elif sz == 224:
            val_bs = max(bs, 256)
        else:
            val_bs = max(bs, 128)
        return dataloader.get_loaders(trndir, valdir, bs=bs, val_bs=val_bs, sz=sz, workers=self.args.workers,
                                      fp16=self.args.fp16, distributed=self.args.distributed, **kwargs)
class DataManager_CIFAR():
    def __init__(self, phases, args):
        print(phases[0])
        self.dataset = datasets.CIFAR10 if args.dataset=='cifar10' else datasets.CIFAR100
        self.cifar_mean = [ 0.485, 0.456, 0.406 ]
        self.cifar_std = [ 0.229, 0.224, 0.225 ]
        self.normalize = transforms.Normalize( mean=self.cifar_mean ,std=self.cifar_std )
        self.args = args
        self.preload_data(**phases[0])

    def set_epoch(self, epoch):
        pass

    def preload_data(self, ep, sz, bs, **kwargs):
        self.trn_dl = torch.utils.data.DataLoader(
            self.dataset(root=self.args.data, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                self.normalize,
            ]), download=True),
            batch_size=bs, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)
        self.val_dl = torch.utils.data.DataLoader(
            self.dataset(root=self.args.data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])),
        batch_size=bs, shuffle=False,
        num_workers=self.args.workers, pin_memory=True)

class DataManager_CINIC10():
    def __init__(self, phases, args):
        print(phases[0])
        self.cinic_mean = [ 0.47889522, 0.47227842, 0.43047404 ]
        self.cinic_std = [ 0.24205776, 0.23828046, 0.25874835 ]
        self.normalize = transforms.Normalize( mean=self.cinic_mean, std=self.cinic_std )
        self.args = args
        self.preload_data(**phases[0])

    def set_epoch(self, epoch):
        pass

    def preload_data(self, ep, sz, bs, **kwargs):
        self.trn_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(self.args.data + '/train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                self.normalize,
            ]) ),
            batch_size=bs, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)
        self.val_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(self.args.data + '/test', transform=transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])),
        batch_size=bs, shuffle=False,
        num_workers=self.args.workers, pin_memory=True)

class DataManager_FLOWER102():
    def __init__( self, phases, args ):
        self.flower_mean = [0.485, 0.456, 0.406]
        self.flower_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize( mean=self.flower_mean, std=self.flower_std )
        self.args = args
        self.preload_data( **phases[0] )
    def set_epoch( self, epoch ):
        pass

    def preload_data( self, ep, sz, bs, **kwargs ):
        self.trn_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(self.args.data + '/train', transform=transforms.Compose([
                transforms.RandomResizedCrop( 224, interpolation=3 ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                self.normalize,
            ]) ),
            batch_size=bs, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)
        self.val_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(self.args.data + '/valid', transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ])),
        batch_size=bs, shuffle=False,
        num_workers=self.args.workers, pin_memory=True)