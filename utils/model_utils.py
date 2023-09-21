import torch
import torch.nn as nn


def get_grad_norm( model ):
	gn = 0
	for p in model.parameters():
		if p.grad is None:
			continue
		g_flat = p.grad.data.view( -1 )
		gn += torch.dot( g_flat, g_flat )

	return torch.sqrt( gn )


def get_param_norm(model):
	pn = 0
	for p in model.parameters():
		if p.grad is None:
			continue
		p_flat = p.data.view( -1 )
		pn += torch.dot( p_flat, p_flat )

	return torch.sqrt( pn )


def get_pblocks( model_params ):
	"""Each Hessian block has two resblocks
	ichnl_per_block = [ 64, 64, 128, 256, 0 ]
	ochnl_per_block = [ 64, 128, 256, 512, 0 ]
	"""

	"""Each Hessian block has four resblocks"""
	ichnl_per_block = [ 64, 0 ]
	ochnl_per_block = [ 64, 0 ]

	b = 0
	co_cur, ci_cur = ochnl_per_block[ 0 ], ichnl_per_block[ 0 ]
	co_nxt, ci_nxt = ochnl_per_block[ 0 ], ichnl_per_block[ 0 ]
	pblocks = {}
	for p in model_params:
		if len( p.shape ) == 1:  # batch norm or bias
			if str( co_cur ) in pblocks:
				pblocks[ str( co_cur ) ].append( p )
			continue
		if len( p.shape ) == 2:  # ignore fc layers
			break
		co, ci = p.shape[ 0 ], p.shape[ 1 ]
		if ci == 3:
			continue
		if co == co_nxt and ci == ci_nxt:
			pblocks[ str( co ) ] = [ p ]
			co_cur, ci_cur = co_nxt, ci_nxt
			b = b + 1 if b < len( ochnl_per_block ) - 1 else b
			co_nxt = ochnl_per_block[ b ]
			ci_nxt = ichnl_per_block[ b ]
		else:
			if str( co_cur ) in pblocks:
				pblocks[ str( co_cur ) ].append( p )

	"""Debug block division
	for key in pblocks.keys():
		print( 'block', key )
		for p in pblocks[ key ]:
			print( p.shape )
	quit()
	"""

	return pblocks


def get_pblocks_resnet50( model_params ):
	ichnl_per_block = [ 64, 256, 512, 1024, 0 ]
	ochnl_per_block = [ 64, 128, 256, 512, 0 ]
	b = 0
	co_cur, ci_cur = ochnl_per_block[ 0 ], ichnl_per_block[ 0 ]
	co_nxt, ci_nxt = ochnl_per_block[ 0 ], ichnl_per_block[ 0 ]
	pblocks = {}
	for p in model_params:
		if len( p.shape ) == 1:  # batch norm or bias
			if str( co_cur ) in pblocks:
				pblocks[ str( co_cur ) ].append( p )
			continue
		if len( p.shape ) == 2:  # ignore fc layers
			break
		co, ci = p.shape[ 0 ], p.shape[ 1 ]
		if ci == 3:
			continue
		if co == co_nxt and ci == ci_nxt:
			pblocks[ str( co ) ] = [ p ]
			co_cur, ci_cur = co_nxt, ci_nxt
			b = b + 1 if b < len( ochnl_per_block ) - 1 else b
			co_nxt = ochnl_per_block[ b ]
			ci_nxt = ichnl_per_block[ b ]
		else:
			if str( co_cur ) in pblocks:
				pblocks[ str( co_cur ) ].append( p )

	return pblocks


def get_pblocks_deit( model_params ):
	pblocks = {}
	for p in model_params:
		if 'deit' not in pblocks:
			pblocks[ 'deit' ] = [ p ]
		else:
			pblocks[ 'deit' ].append( p )

	"""debug block division
	for key in pblocks.keys():
		print( 'block', key )
		for p in pblocks[ key ]:
			print( p.shape )
	quit()
	"""

	return pblocks


from opt.slim import SlimQN
from opt.slimblock import BlockSlimQN
# import kfac
from opt.sgd import SGDOptimizer


def get_optimizer( args, model, params, pblocks ):
	optimizer, preconditioner, kfac_param_scheduler = None, None, None
	if args.optimizer == 'SGD':
		preconditioner = None
		optimizer = SGDOptimizer(
			params,
			lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
		)
	elif args.optimizer == 'ADAM':
		preconditioner = None
		optimizer = torch.optim.AdamW(
			params,
			lr=args.lr,
			betas=( 0.9, 0.999 ), weight_decay=args.weight_decay
		)
	elif args.optimizer == 'SLIMBLOCK':
		preconditioner = None
		optimizer = BlockSlimQN(
			params, pblocks,
			lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
			mm_p=args.mm_p, mm_g=args.mm_g, mmm=args.mmm, update_freq=args.update_freq,
			hist_sz=args.hist_sz, damping=args.lbfgs_damping, kl_clip=args.grad_clip
		)
	elif args.optimizer == 'SLIM':
		preconditioner = None
		optimizer = SlimQN(
			params,
			lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
			mm_p=args.mm_p, mm_g=args.mm_g, update_freq=args.update_freq,
			hist_sz=args.history_size, damping=args.lbfgs_damping, kl_clip=args.grad_clip
		)
	elif args.optimizer == 'KFAC':
		if args.kfac_comm_method == 'comm-opt':
			comm_method = kfac.CommMethod.COMM_OPT
		elif args.kfac_comm_method == 'mem-opt':
			comm_method = kfac.CommMethod.MEM_OPT
		elif args.kfac_comm_method == 'hybrid-opt':
			comm_method = kfac.CommMethod.HYBRID_OPT
		else:
			raise ValueError('Unknwon KFAC Comm Method: {}'.format(
				args.kfac_comm_method))
		preconditioner = kfac.KFAC(
			model,
			damping=args.damping,
			factor_decay=args.stat_decay,
			factor_update_freq=args.kfac_cov_update_freq,
			inv_update_freq=args.kfac_update_freq,
			kl_clip=args.kl_clip,
			lr=args.lr,
			batch_first=True,
			comm_method=comm_method,
			distribute_layer_factors=not args.coallocate_layer_factors,
			grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
			grad_worker_fraction=args.kfac_grad_worker_fraction,
			skip_layers=args.skip_layers,
			use_eigen_decomp=not args.use_inv_kfac,
		)
		kfac_param_scheduler = kfac.KFACParamScheduler(
			preconditioner,
			damping_alpha=args.damping_alpha,
			damping_schedule=args.damping_decay,
			update_freq_alpha=args.kfac_update_freq_alpha,
			update_freq_schedule=args.kfac_update_freq_decay
		)
		optimizer = SGDOptimizer(
			params,
			lr=args.lr,
			momentum=args.momentum,
			weight_decay=args.weight_decay
		)
	else:
		print('[ERROR] choose a valid optmizer!')
		quit()

	return optimizer, preconditioner, kfac_param_scheduler
