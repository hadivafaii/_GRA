import torch


class RelativisticLion(torch.optim.Optimizer):
	"""
	Relativistic Lion Optimizer.

	Derived from relativistic Lagrangian mechanics with gauge coupling
	and relativistic Rayleigh dissipation.

	Args:
		params: Model parameters
		lr: Learning rate η (physical interpretation: η = c·Δt)
		betas: (β1, β2) where β1 controls interpolation, β2 controls momentum EMA
		weight_decay: Decoupled weight decay coefficient
		mass: Rest mass of the particle. Controls smoothness of saturation.
			mass=0: Exact sign() operation (standard Lion, massless limit)
			mass>0: Smooth saturation (Massive Lion)
		use_curvature: Whether to apply curvature correction to momentum

	Physical correspondence (with Δt = 1):
		- c = η (speed of light = learning rate)
		- τ = 1/(1-β₂) (dissipation timescale)
		- k = (1-β₁)/β₁ (gauge coupling)
		- ρ = m·c/τ = m·η·(1-β₂) (effective mass scale in velocity equation)

	The velocity update is:
		v = c_t / √(c_t² + ρ²)

	which approaches sign(c_t) as ρ → 0 and becomes linear as ρ → ∞.
	"""

	def __init__(
			self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0,
			mass=0.0, use_curvature=False
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 < betas[0] < 1.0:
			raise ValueError(f"Invalid beta1: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta2: {betas[1]}")
		if mass < 0:
			raise ValueError(f"Invalid mass: {mass}")

		defaults = dict(
			lr=lr,
			betas=betas,
			weight_decay=weight_decay,
			mass=mass,
			use_curvature=use_curvature,
			# rho=mass * lr * (1 - betas[1]),
			# kappa=(1 - betas[0]) / betas[0],
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr = group['lr']
			beta1, beta2 = group['betas']
			wd = group['weight_decay']
			mass = group['mass']
			use_curvature = group['use_curvature']
			# handle rho & kappa
			rho = mass * lr * (1 - beta2)
			kappa = (1 - beta1) / beta1

			for p in group['params']:
				if p.grad is None:
					continue

				grad = p.grad
				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(
						p, memory_format=torch.preserve_format)
					if use_curvature:
						state['prev_grad'] = torch.zeros_like(
							p, memory_format=torch.preserve_format)

				state['step'] += 1
				exp_avg = state['exp_avg']

				# 1. Decoupled Weight Decay
				if wd > 0:
					p.mul_(1 - lr * wd)

				# 2. Interpolation: c_t = β₁·m_{t-1} + (1-β₁)·g_t
				c_t = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
				# c_t = exp_avg.clone().mul_(beta1).add_(
				# 	grad, alpha=1 - beta1)

				# 3. Relativistic Velocity: v = c_t / √(c_t² + ρ²)
				if mass == 0:
					update = torch.sign(c_t)
				else:
					rho_t = c_t.new_tensor(rho)
					update = c_t / torch.hypot(c_t, rho_t)
					# update = c_t / torch.sqrt(c_t.square() + rho ** 2)

				# 4. Parameter Update: θ_t = θ_{t-1} - η·v_t
				p.add_(update, alpha=-lr)

				# 5. Momentum Update: m_t = β₂·m_{t-1} + (1-β₂)·g_t
				exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

				# 6. Curvature Correction: m_t -= κ·(g_t - g_{t-1})
				if use_curvature and state['step'] > 1:
					exp_avg.add_(
						grad - state['prev_grad'],
						alpha=-kappa,
					)

				# 7. Store gradient for next curvature computation
				if use_curvature:
					state['prev_grad'].copy_(grad)

		return loss
