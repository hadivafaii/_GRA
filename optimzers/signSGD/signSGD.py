"""
	Custom implementatio of signSGD and signum.
	Adapted from: https://github.com/Niccolo-Ajroldi/plainLM
"""

import torch


class SignSGD(torch.optim.Optimizer):
	def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0):
		if not 0.0 <= lr:
			raise ValueError(f'Invaid learing rate: {lr}')
		if not 0.0 <= momentum or not momentum <= 1.0:
			raise ValueError(f'Invaid momentum: {momentum}')
		if not 0.0 <= dampening or not dampening <= 1.0:
			raise ValueError(f'Invaid dampening: {dampening}')
		if not 0.0 <= weight_decay:
			raise ValueError(f'Invaid weight decay: {weight_decay}')

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
		super(SignSGD, self).__init__(params, defaults)

	@torch.no_grad()
	def step(self):
		for group in self.param_groups:
			alpha = group['lr']
			momentum = group['momentum']
			dampening = group['dampening']
			weight_decay = group['weight_decay']

			for p in group['params']:
				if p.grad is None:
					continue

				param_state = self.state[p]

				# m initialization
				if 'm' not in param_state:
					param_state['m'] = p.grad.detach().clone()

				# weight decay
				if weight_decay > 0:
					p.mul_(1 - alpha * weight_decay)

				# decay momentum
				m = param_state['m']
				m.mul_(momentum).add_(p.grad, alpha=(1.0 - dampening))

				# update p
				p.add_(torch.sign(m), alpha=-alpha)
