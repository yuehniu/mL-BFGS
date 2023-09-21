"""Block-wise SLIM-QN
Desc:
    Block-wise SLIM-QN that perform SLIM-QN on blocks,
    where each block may consists of several layers.

Author:

Note:
"""
import torch
import torch.optim as optim
import math

epsilon = 1e-12


class BlockSlimQN( optim.Optimizer ):
    def __init__(
            self, model_parameters, blocks: dict,
            lr=1, momentum=0.9, weight_decay=0.0,
            mm_p=0.9, mm_g=0.99, mmm=0.8,
            update_freq=100, hist_sz=100, decay_period=10, damping=0.2, kl_clip=0.005
    ):
        """
        Args:
            model_parameters: DNN model parameters
            blocks: block configuration, each block has several layers
            lr: learning rate
            momentum: momentum for model update
            weight_decay: weight decay
            mm_p: momentum for averaging history param
            mm_g: momentum for averaging hitory grad
            mmm: double momentum for average dg and dp
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
            mm_p=mm_p, mm_g=mm_g, mmm=mmm, damping=damping, kl_clip=kl_clip,
            update_freq=update_freq, hist_sz=hist_sz
        )
        super( BlockSlimQN, self ).__init__( model_parameters, defaults )

        self.blocks = blocks

        # history vectors (dg, dp) and rho
        self.hist_dg, self.hist_dp, self.rho_list = {}, {}, {}

        # internal variables for deriving (dg, dp) and rho
        self.avg_p, self.hist_avg_p = {}, {}
        self.avg_g, self.hist_avg_g = {}, {}
        self.has_avg_p, self.has_avg_g, self.has_hist_p, self.has_hist_g = {}, {}, {}, {}

        # initialzing based on blocks from inputs
        for bk in blocks.keys():
            self.hist_dg[ bk ], self.hist_dp[ bk ], self.rho_list[ bk ] = [], [], []
            self.avg_p[ bk ], self.hist_avg_p[ bk ] = [], []
            self.avg_g[ bk ], self.hist_avg_g[ bk ] = [], []
            self.has_avg_p[ bk ], self.has_avg_g[ bk ] = False, False
            self.has_hist_p[ bk ], self.has_hist_g[ bk ] = False, False

        self.tao_lb = 0.01
        self.tao_ub = 1.5
        self.tao_scale = 1

        # iteration info
        self.init_lr = lr
        self.steps = 0  # batch-wise
        self.update_dg_dp = False
        self.epoch = 0
        self.start_slim = False
        self.decay_period = decay_period
        self.h0 = {}

        # some data for debug purpose (block-wise)
        self.tao_before, self.tao_after = {}, {}
        self.gn_before, self.gn_after = {}, {}
        self.avg_pn, self.avg_gn = {}, {}
        self.pn = 0.0  # parameter norm

        for bk in blocks.keys():
            self.tao_before[ bk ], self.tao_after[ bk ] = 1.0, 1.0
            self.gn_before[ bk ], self.gn_before[ bk ] = 0.0, 0.0
            self.avg_pn[ bk ], self.avg_gn[ bk ] = 0.0, 0.0

        print("[BLOCKSLIM] initialize BlockSlimQN optimizer:\n"
              "-----------------------------------------------\n"
              f"\tInitial learning rate: {lr}\n"
              f"\tMomentum for update: {momentum}\n"
              f"\tWeight decay: {weight_decay}\n"
              f"\tDamping factor: {damping}\n"
              f"\tMomentum for param: {mm_p}\n"
              f"\tMomentum for grad: {mm_g}\n"
              f"\tDecay period: {decay_period}\n"
              f"\tHistory vector size: {hist_sz}\n"
              f"\tBase Hessian update frequency: {update_freq}\n"
              f"\tNumber of threads: {torch.get_num_threads()}\n"
              "-----------------------------------------------")

    @staticmethod
    def __flatten(tensorlist):
        views = []
        for p in tensorlist:
            view = p.view( -1 )
            views.append( view )
        return torch.cat( views, 0 )

    @staticmethod
    def __inv_flatten( vec, refparam ):
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
        """ one way
        pdata_avg *= stat_decay / ( 1 - stat_decay )
        pdata_avg += pdata
        pdata_avg *= ( 1 - stat_decay )
        """

        """another way"""
        pdata_avg.mul_( stat_decay ).add_( pdata, alpha=1.0 )

    def __get_block_key( self, p ):
        """
        get corresponding key by comparing p with each element in blocks
        @param p: param in certain layer
        @return:
        """
        find_key = False
        for bk in self.blocks.keys():
            for pref in self.blocks[ bk ]:
                if p is pref:
                    find_key = True
                    return bk
        # assert find_key is True, 'key does not exists in parameter blocks'
        return None

    def __get_dp( self ):
        """
        get momentum of parameter in every iteration,
        then compute dp if the Hessian update is needed
        @return:
        """
        for pgroup in self.param_groups:
            dp = {}
            for bk in self.blocks.keys():
                dp[ bk ] = []

            # - get history smoothed parameters with momentum
            i = 0
            bk, bk_prev = '', ''
            mm_p, mmm = pgroup[ 'mm_p' ], pgroup[ 'mmm' ]
            lr = pgroup[ 'lr' ]
            for p in pgroup[ 'params' ]:
                if p.grad is None:
                    continue
                bk = self.__get_block_key( p )
                if bk is None:
                    continue
                i = 0 if bk != bk_prev else i+1
                if not self.has_avg_p[ bk ]:
                    self.avg_p[ bk ].append( p.data.clone() )
                else:
                    self.__update_avg( p.data, self.avg_p[ bk ][ i ], mm_p )
                    if self.update_dg_dp:
                        if self.has_hist_p[ bk ]:
                            dp[ bk ].append( self.avg_p[ bk ][ i ] - self.hist_avg_p[ bk ][ i ] )
                            self.hist_avg_p[ bk ][ i ].copy_( self.avg_p[ bk ][ i ] )
                        else:
                            self.hist_avg_p[ bk ].append( self.avg_p[ bk ][i].clone() )
                        # self.avg_p[ bk ][ i ].copy_( p.data )
                bk_prev = bk

            for bk in self.blocks.keys():
                self.has_avg_p[ bk ] = True
                self.has_hist_p[ bk ] = len( self.hist_avg_p[ bk ] ) > 0

            # - update hist_dp
            # - remove old dp is hist_dp is full
            hist_sz = pgroup['hist_sz']
            for bk in self.blocks.keys():

                if len( dp[ bk ] ) > 0:
                    l = len( self.hist_dp[ bk ] )
                    dp_flatten = self.__flatten( dp[ bk ] )
                    # dp_flatten.div_( 10 * self.avg_pn[ bk ] )

                    """double momentum"""
                    if l > 0:
                        dp_flatten.mul_( 1-mmm ).add_( self.hist_dp[ bk ][-1], alpha=mmm )

                    self.avg_pn[ bk ] = torch.norm( dp_flatten )

                    if l == hist_sz:
                        dp_old = self.hist_dp[ bk ].pop(0)
                        del dp_old
                    self.hist_dp[bk].append( dp_flatten.detach() )

    def __get_dg( self ):
        """
        get momentum of gradients in every iteration,
        then compute dg if the Hessian update is needed
        @return:
        """
        for pgroup in self.param_groups:
            dg = {}
            for bk in self.blocks.keys():
                dg[ bk ] = []

            # - get history smoothed gradients with momentum
            i = 0
            bk, bk_prev = '', ''
            lr, wd = pgroup[ 'lr' ], pgroup[ 'weight_decay' ]
            mm_g, mmm = pgroup[ 'mm_g' ], pgroup[ 'mmm' ]
            for p in pgroup[ 'params' ]:
                if p.grad is None:
                    continue
                bk = self.__get_block_key( p )
                if bk is None:
                    continue
                i = 0 if bk != bk_prev else i+1
                g = p.grad.data.clone()
                g.add_( g, alpha=wd )
                if not self.has_avg_g[ bk ]:
                    self.avg_g[ bk ].append( g )
                else:
                    self.__update_avg( g, self.avg_g[ bk ][ i ], mm_g )
                    if self.update_dg_dp:
                        if self.has_hist_g[ bk ]:
                            dg[ bk ].append( self.avg_g[ bk ][i] - self.hist_avg_g[ bk ][i] )
                            self.hist_avg_g[ bk ][ i ].copy_( self.avg_g[ bk ][ i ] )
                        else:
                            self.hist_avg_g[ bk ].append( self.avg_g[ bk ][i].clone() )
                        self.avg_g[ bk ][ i ].copy_( g.data )
                bk_prev = bk

            for bk in self.blocks.keys():
                self.has_avg_g[ bk ] = True
                self.has_hist_g[ bk ] = len( self.hist_avg_g[ bk  ] ) > 0

            # - update hist_dg
            # - add damping
            # - remove old dg is hist_dg is full
            for bk in self.blocks.keys():
                scaling, hist_sz = lr / self.init_lr, pgroup[ 'hist_sz' ]
                if len( dg[ bk ] ) > 0:
                    l = len( self.hist_dg[ bk ] )
                    dg_flatten = self.__flatten( dg[ bk ] )
                    # dg_flatten.div_( 10* self.avg_gn[ bk ] )
                    # dg_flatten.mul_( scaling )

                    """double momentum"""
                    if l > 0:
                        dg_flatten.mul_( 1-mmm ).add_( self.hist_dg[ bk ][-1], alpha=mmm )

                    self.avg_gn[ bk ] = torch.norm( dg_flatten )

                    phi = 1.0
                    damping = pgroup[ 'damping' ]
                    s = self.hist_dp[ bk ][ -1 ]
                    y = dg_flatten

                    self.tao_before[ bk ] = torch.dot( s, y ) / ( torch.dot( s, s ) + epsilon )
                    if self.tao_before[ bk ] < self.tao_lb:
                        phi = ( self.tao_scale - self.tao_lb ) / ( self.tao_scale - self.tao_before[ bk ] )
                    elif self.tao_before[ bk ] > self.tao_ub:
                        phi = ( self.tao_ub - self.tao_scale ) / ( self.tao_before[ bk ] - self.tao_scale )
                    phi = min( phi, 1.0 - damping )
                    # add damping, equivalently to add weight decay
                    dg_flatten.mul_( phi ).add_( s, alpha=self.tao_scale * (1-phi) )
                    self.tao_after[ bk ] = torch.dot( s, dg_flatten ) / ( torch.dot(s, s) + epsilon )

                    if l == hist_sz:
                        dg_old = self.hist_dg[ bk ].pop( 0 )
                        del dg_old
                        rho_old = self.rho_list[ bk ].pop(0)
                        del rho_old
                    self.hist_dg[ bk ].append( dg_flatten )
                    self.__get_rho( bk )

    def __get_rho( self, bk ):
        assert len( self.hist_dp[ bk ] ) == len( self.hist_dg[ bk ] ), 'dg and dg have different length'
        rho = torch.dot( self.hist_dp[ bk ][ -1 ], self.hist_dg[ bk ][ -1 ] )

        self.rho_list[ bk ].append(rho)
        self.start_slim = True
        self.h0[ bk ] = self.rho_list[ bk ][-1] / \
            ( torch.dot( self.hist_dg[ bk ][-1], self.hist_dg[ bk ][-1] ) + epsilon )

    def __grad_cond( self ):
        """
        gradient conditioning block by block
        @return:
        """
        for pgroup in self.param_groups:
            plist, glist = {}, {}
            for bk in self.blocks.keys():
                plist[ bk ], glist[ bk ] = [], []

            # - group param/gradients into each block
            gnorm = 0.0
            for p in pgroup[ 'params' ]:
                bk = self.__get_block_key( p )
                if p.grad is None:
                    continue
                if bk is None:
                    continue
                plist[ bk ].append( p.data )
                glist[ bk ].append( p.grad.data )

            # - flatten each group into a vector
            # - apply hessian-vector product
            for bk in self.blocks.keys():
                g_flat = self.__flatten( glist[ bk ] )
                p_flat = self.__flatten( plist[ bk ] )
                wd = pgroup[ 'weight_decay' ]
                g_flat.add_( p_flat, alpha=wd )
                g = g_flat.clone()

                # - some debugging info
                self.gn_before[ bk ] = torch.sqrt( torch.dot( g, g ) )
                self.pn += torch.sqrt( torch.dot( p_flat, p_flat ) )

                l = len( self.hist_dp[ bk ] )
                alpha_list = []
                for i in range( 0, l ):
                    alpha = torch.dot( self.hist_dp[ bk ][ l-1-i ], g ) / ( self.rho_list[ bk ][ l-1-i ] )
                    alpha_list.append( alpha )
                    g.add_( self.hist_dg[ bk ][ l-1-i ], alpha=-alpha )
                g.mul_( self.h0[ bk ] )
                for i in range( l, 0, -1 ):
                    beta = torch.dot( self.hist_dg[ bk ][ l-i ], g ) / ( self.rho_list[ bk ][ l-i ] )
                    g.add_( self.hist_dp[ bk ][l-i], alpha=alpha_list[ i-1 ]-beta )

                # apply KL-clipping
                # kl_clip = pgroup[ 'kl_clip' ]
                # vg_sum = torch.dot( g_flat, g )
                # nu = min( 1.0, math.sqrt( kl_clip / vg_sum ) )
                # normalize gradients
                gnorm += torch.dot( g, g )
                self.gn_after[ bk ] = torch.sqrt( torch.dot( g, g ) )
                # self.gn_after[ bk ].mul_( nu )
                g_shaped = self.__inv_flatten( g, self.blocks[ bk ] )
                for p, g_p in zip( self.blocks[ bk ], g_shaped ):
                    if p.grad is None:
                        continue
                    p.grad.data.copy_( g_p )
                    # p.grad.data.mul_( nu )

            for p in pgroup[ 'params' ]:
                if p.grad is None:
                    continue
                p.grad.data.div_( torch.sqrt( gnorm ) )

    @torch.no_grad()
    def step(self, closure=None, epoch=0 ):
        update_freq = self.param_groups[ 0 ][ 'update_freq' ]

        self.steps += 1
        self.epoch = epoch
        self.update_dg_dp = ( self.steps % update_freq == 0 )

        self.__get_dp()
        self.__get_dg()

        if self.start_slim:
            self.pn = 0.0
            self.__grad_cond()

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
