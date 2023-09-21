import math

import torch
import torch.optim as optim

class SGDOptimizer(optim.Optimizer):
    def __init__(self,
                 model_parameters,
                 lr=0.001,
                 momentum=0.9,
                 damping = 1.0,
                 weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(SGDOptimizer, self).__init__(model_parameters, defaults)

        # for debug purpose
        self.gn = 0.0
        self.pn = 0.0
        self.steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        self.gn = 0.0
        self.pn = 0.0
        self.steps += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            damping = group['damping']

            for p in group['params']:
                if p.grad is None:
                    continue
                p_flat = p.data.view(-1)
                self.pn += torch.dot(p_flat, p_flat)
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                g_flat = d_p.view(-1)
                self.gn += torch.dot(g_flat, g_flat)
            self.pn = torch.sqrt(self.pn)
            self.gn = torch.sqrt(self.gn)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p.div_( self.gn )
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(damping, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

