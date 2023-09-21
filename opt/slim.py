"""Stochastic LBFGS optmizer with Damping
DESC:
    A naive distributed version of Stochastic LBFGS implementation.
    The main issue in distributing LBFGS is how to store/access two history vars:
    self.hist_dg and self.hist_dp.
    In the naive implementation, we split these two vars and assigned to four GPUS.
    When accessing the vars, we need to explicitely move them from one GPU to another.

AUTHOR:

NOTE:
"""
import torch
import torch.optim as optim
import math

epsilon = 0.00001


class SlimQN( optim.Optimizer ):
    def __init__(
            self, model_parameters,
            lr=1, momentum=0.9, weight_decay=0.0, rho_min=0.0001,
            mm_p=0.9, mm_g=0.99, update_freq=100, hist_sz=100,
            decay_period=10, damping=0.2, kl_clip=0.005
    ):
        """
        Args:
            model_parameters: DNN model parameters
            lr: learning rate
            momentum: momentum for model update
            weight_decay: weight decay
            rho_min: a threshold to decide whether to store history vector
            mm_p: momentum for averaging history param
            mm_g: momentum for averaging hitory grad
            update_freq: frequency for updating Hessian Inverse
            hist_sz: size of history vectors
            decay_period: learning rate decay period
            damping: damping factor
            kl_clip: KL clipping factor
        """ 
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            mm_p=mm_p, mm_g=mm_g, damping=damping, kl_clip=kl_clip,
            update_freq=update_freq, hist_sz=hist_sz
        )
        super( SlimQN, self ).__init__( model_parameters, defaults )

        # - history vectors
        self.hist_dg, self.hist_dp, self.rho_list = [], [], []

        # use avg_var to calculcate self.history_delta_var
        self.avg_p,  self.hist_avg_p = [], []
        self.avg_g, self.hist_avg_g = [], []
        self.has_avg_p, self.has_avg_g = False, False
        self.has_hist_p, self.has_hist_g = False, False

        self.rho_min = rho_min
        self.tao_lb, self.tao_ub = 0.01, 1.0

        self.model_param = self.param_groups[0]['params']

        # iteration info
        self.init_lr = lr
        self.steps = 0  # batch-wise
        self.update_dg_dp = False
        self.epoch = 0
        self.start_slim = False
        self.decay_period = decay_period
        self.h0 = 0

        # some data for debug purpose
        self.tao_before, self.tao_after = 1.0, 1.0
        self.gn_before, self.gn_after = 0.0, 0.0  # g norm before/after conditioning
        self.pn = 0.0  # p norm

        print("[SLIM] initialize SlimQN optimizer:\n"
              "-------------------------------------\n"
              f"\tInitial learning rate: {lr}\n"
              f"\tMomentum for update: {momentum}\n"
              f"\tWeight decay: {weight_decay}\n"
              f"\tDamping factor: {damping}\n"
              f"\tMomentum for param: {mm_p}\n"
              f"\tMomentum for grad: {mm_g}\n"
              f"\tHistory vector size: {hist_sz}\n"
              f"\tBase Hessian update frequency: {update_freq}\n"
              f"\tGradient clipping: {kl_clip}\n"
              f"\tNumber of threads: {torch.get_num_threads()}\n"
              "-------------------------------------")

    @staticmethod
    def __flattern( tensorlist, grad=False ):
        views = []
        for p in tensorlist:
            if grad:
                view = p.grad.view( -1 )
            else:
                view = p.view( -1 )
            views.append( view )
        return torch.cat( views, 0 )

    @staticmethod
    def __inv_flattern( vec, refparam ):
        offset = 0
        views = []
        for p in refparam:
            if p.grad is None:
                continue
            tmp = vec[ offset:offset+p.data.numel() ]
            view = tmp.view( p.data.size() )
            views.append( view )
            offset += p.data.numel()
        return views

    @staticmethod
    def __update_avg( pdata, pdata_avg, stat_decay ):
        pdata_avg *= stat_decay / ( 1 - stat_decay )
        pdata_avg += pdata
        pdata_avg *= ( 1 - stat_decay )

    def __get_dp(self):
        i = 0
        dp = []
        mm_p = self.param_groups[ 0 ][ 'mm_p' ]
        for p in self.model_param:
            if p.grad is None: 
                continue
            if not self.has_avg_p:
                self.avg_p.append( p.data.clone() )
            else:
                self.__update_avg( p.data, self.avg_p[i], mm_p )
                if self.update_dg_dp:
                    if self.has_hist_p:
                        dp.append( self.avg_p[i] - self.hist_avg_p[i] )
                        self.hist_avg_p[i].copy_( self.avg_p[i] )
                    else:
                        self.hist_avg_p.append( self.avg_p[i].clone() )
                    #self.avg_p[i] = p.data.clone()
            i += 1
        self.has_avg_p = True
        self.has_hist_p = len( self.hist_avg_p ) > 0

        # - update hist_dp
        # - remove old dp is hist_dp is full
        hist_sz = self.param_groups[0]['hist_sz']
        if len( dp ) > 0:
            l = len(self.hist_dp)
            dp_flatten = self.__flattern( dp )
            if l == hist_sz:
                dp_old = self.hist_dp.pop(0)
                del dp_old
            self.hist_dp.append( dp_flatten.detach() )

    def __get_dg(self):
        i = 0
        dg = []
        lr = self.param_groups[ 0 ][ 'lr' ]
        mm_g = self.param_groups[ 0 ][ 'mm_g' ]
        for p in self.model_param:
            if p.grad is None:
                continue
            g = p.grad.data.clone()
            if not self.has_avg_g:
                self.avg_g.append(g)
            else:
                self.__update_avg( g, self.avg_g[i], mm_g )
                if self.update_dg_dp:
                    if self.has_hist_g:
                        dg.append( self.avg_g[i] - self.hist_avg_g[i] )
                        self.hist_avg_g[i].copy_( self.avg_g[i] )
                    else:
                        self.hist_avg_g.append( self.avg_g[i].clone() )
                    # self.avg_g[i] = g.clone()
            i += 1
        self.has_avg_g = True
        self.has_hist_g = len( self.hist_avg_g ) > 0

        # - update hist_dg
        # - add damping
        # - remove old dg is hist_dg is full
        scaling = lr / self.init_lr
        hist_sz = self.param_groups[ 0 ][ 'hist_sz' ]
        if len( dg ) > 0:
            l = len( self.hist_dg )
            dg_flatten = self.__flattern( dg )
            # dg_flatten.mul_( scaling )  for naive L-BFGS

            # add damping
            damping = self.param_groups[ 0 ][ 'damping' ]
            s = self.hist_dp[ -1 ]
            if damping > 0.0:
                phi = 1.0
                y = dg_flatten
                self.tao_before = torch.dot( s, y ) / ( torch.dot( s, s ) + epsilon )
                if self.tao_before < self.tao_lb:
                    phi = ( 1.0 - self.tao_lb ) / ( 1.0 - self.tao_before )
                elif self.tao_before > self.tao_ub:
                    phi = ( self.tao_ub - 1 ) / ( self.tao_before - 1.0 )
                phi = min( phi, 1 - damping )
                dg_flatten.mul_( phi ).add_( s, alpha=1 - phi )
            self.tao_after = torch.dot( s, dg_flatten ) / ( torch.dot(s, s) + epsilon )

            if l == hist_sz:
                dg_old = self.hist_dg.pop(0)
                del dg_old
                rho_old = self.rho_list.pop(0)
                del rho_old
            # move dg to cpu memory
            self.hist_dg.append(dg_flatten)
            self.__get_rho()

    def __get_rho( self ):
        assert len( self.hist_dp ) == len( self.hist_dg )
        rho = torch.dot( self.hist_dp[ -1 ], self.hist_dg[ -1 ] )
        if rho < self.rho_min:
            dp_bad = self.hist_dp.pop(-1)
            del dp_bad
            dg_bad = self.hist_dg.pop(-1)
            del dg_bad
            # mdg_bad = self.hist_mdg.pop(-1)
            # del mdg_bad
            return

        self.rho_list.append(rho)
        self.start_slim = True
        self.h0 = self.rho_list[-1] / ( torch.dot( self.hist_dg[-1], self.hist_dg[-1] ) + epsilon )

    def __update_gradient( self ):
        l = len( self.hist_dp )
        assert l == len( self.hist_dg )
        assert l == len( self.rho_list )
        wd = self.param_groups[ 0 ][ 'weight_decay' ]
        g_flat = self.__flattern( self.model_param, grad=True )
        p_flat = self.__flattern( self.model_param )
        g = torch.add( g_flat, p_flat, alpha=wd )

        self.gn_before = torch.sqrt( torch.dot( g, g ) )
        self.pn = torch.sqrt( torch.dot( p_flat, p_flat ) )

        # Hessian-vector product
        alpha_list = [] 
        for i in range( 0, l ):
            alpha = torch.dot( self.hist_dp[l-1-i], g ) / ( self.rho_list[l-1-i] + epsilon )
            alpha_list.append( alpha )
            g.add_( self.hist_dg[l-1-i], alpha=-alpha )
        g.mul_( self.h0 )  # for naive L-BFGS
        for i in range( l, 0, -1 ):
            beta = torch.dot( self.hist_dg[l-i], g ) / ( self.rho_list[l-i] + epsilon )
            g.add_( self.hist_dp[l-i], alpha=alpha_list[i-1]-beta )

        # apply KL-clipping
        kl_clip = self.param_groups[ 0 ][ 'kl_clip' ]
        if kl_clip > 0.0:
            vg_sum = torch.dot( g_flat, g )
            nu = min( 1.0, math.sqrt( kl_clip / vg_sum ) )
        else:
            nu = 1.0
        self.gn_after = torch.sqrt( torch.dot( g, g ) )
        self.gn_after.mul_( nu )
        g_shaped = self.__inv_flattern( g, self.model_param )
        for p, g_p in zip( self.model_param, g_shaped ):
            if p.grad is None: 
                continue
            p.grad.data.copy_( g_p )
            p.grad.data.mul_( nu )

    @torch.no_grad()
    def step(self, closure=None, epoch=0 ):
        update_freq = self.param_groups[ 0 ][ 'update_freq' ]

        self.steps += 1
        self.epoch = epoch
        self.update_dg_dp = ( self.steps % update_freq == 0 )
        # save gradients and param for gradient update

        self.__get_dp()
        self.__get_dg()

        if self.start_slim:
            self.__update_gradient()

        for group in self.param_groups:
            momentum = group[ 'momentum' ]
            wd = group[ 'weight_decay' ]
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad.data
                # warm-up stage (SGD)
                if not self.start_slim:
                    dp.add_( p.data, alpha=wd )
                if momentum != 0:
                    param_state = self.state[ p ]
                    if 'momentum_buf' not in param_state:
                        buf = param_state[ 'momentum_buf' ] = torch.zeros_like( p.data )
                        buf.mul_( momentum ).add_( dp )
                    else:
                        buf = param_state['momentum_buf']
                        buf.mul_( momentum ).add_( 1.0, dp )
                    dp = buf
                p.data.add_( -group[ 'lr' ], dp )
