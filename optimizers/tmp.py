import torch


class RelativisticMemoryOptimizer(torch.optim.Optimizer):
	def __init__(self, base_optimizer, memory_timescale=0.01, coupling_strength=1.0):
		if not isinstance(base_optimizer, torch.optim.Optimizer):
			raise TypeError("base_optimizer must be a torch.optim.Optimizer")

		self.base_optimizer = base_optimizer
		self.lamb = float(coupling_strength)
		self.tau = float(memory_timescale)

		# Proper Optimizer initialization
		defaults = dict(base_optimizer.defaults)
		super().__init__(base_optimizer.param_groups, defaults)

		# Share the exact same param_groups and state with the base optimizer
		self.param_groups = self.base_optimizer.param_groups
		self.state = self.base_optimizer.state

	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		self._update_memory_and_force()
		self.base_optimizer.step()
		return loss

	def zero_grad(self, set_to_none=True):
		self.base_optimizer.zero_grad(set_to_none=set_to_none)

	def state_dict(self):
		return self.base_optimizer.state_dict()

	def load_state_dict(self, state_dict):
		self.base_optimizer.load_state_dict(state_dict)
		self.param_groups = self.base_optimizer.param_groups
		self.state = self.base_optimizer.state

	@torch.no_grad()
	def _update_memory_and_force(self):
		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue
				if p not in self.state:
					continue
				state = self.state[p]
				if len(state) == 0:
					continue

				if "phi_field" not in state:
					state["phi_field"] = p.detach().clone()
					continue

				phi = state["phi_field"]
				# Apply restoring force: grad += lamb * (p - phi)
				p.grad.add_(p - phi, alpha=self.lamb)
				# Update memory: phi <- phi + tau * (p - phi)
				phi.add_(p - phi, alpha=self.tau)


class MagneticSteeringOptimizer(torch.optim.Optimizer):
	"""
	A meta-optimizer that applies the 'Orthogonal Steering Gauge'
	(Synthetic Lorentz Force) to the gradients before passing them
	to the base optimizer.

	Proposal 4 Logic:
	1. Calculate the 'velocity' (momentum) of the learner.
	2. Decompose the current gradient into parallel and
		orthogonal components relative to velocity.
	3. Inject a 'steering force' that amplifies the
		orthogonal gradient component, pushing the
		learner to explore valley walls (high curvature)
		without changing acceleration down the valley floor.

	Args:
		base_optimizer (torch.optim.Optimizer):
			The optimizer to wrap (e.g. AdamW).
		steering_strength (float):
			Lambda parameter. Controls the magnitude of the
			transverse force. Recommended range: 0.1 - 1.0.
		stabilize (bool):
			If True, normalizes the projection by ||v||^2 (Projected Gauge).
			If False, uses the raw Triple Product (Proposal 4 strict).
			Recommended: True (for numerical stability).
	"""

	def __init__(
			self,
			base_optimizer: torch.optim.Optimizer,
			steering_strength: float = 0.5,
			stabilize: bool = True,
	):
		self.base_optimizer = base_optimizer
		self.steering_strength = steering_strength
		self.stabilize = stabilize

		super().__init__(
			self.base_optimizer.param_groups,
			self.base_optimizer.defaults,
		)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		# Apply Steering Force to Gradients
		self._apply_steering()

		# Step the base optimizer using the modified gradients
		self.base_optimizer.step()

		return loss

	def zero_grad(self, set_to_none=True):
		self.base_optimizer.zero_grad(
			set_to_none=set_to_none)

	@torch.no_grad()
	def _apply_steering(self):
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue

				grad = p.grad
				state = self.base_optimizer.state[p]

				# 1. Retrieve Velocity (Momentum)
				# We check for standard buffer names
				# used in PyTorch optimizers
				if 'exp_avg' in state:
					# Adam/AdamW/RMSprop style
					velocity = state['exp_avg']
				elif 'momentum_buffer' in state:
					# SGD style
					velocity = state['momentum_buffer']
				else:
					# If no momentum exists yet (iter 0),
					# steering is undefined.
					continue

				# 2. Compute the Steering Force
				# (Orthogonal Projection)
				# F_steer is proportional to the
				# gradient component orthogonal to velocity.

				# Flatten for dot product
				v_flat = velocity.view(-1)
				g_flat = grad.view(-1)

				v_norm_sq = torch.dot(v_flat, v_flat)

				if v_norm_sq == 0:
					continue

				# Projection of Gradient onto Velocity:
				# proj = (g . v) / ||v||^2 * v
				dot_gv = torch.dot(g_flat, v_flat)

				if self.stabilize:
					# Normalized Projection
					# (Unit consistent with gradient)
					# Force = lambda * (g_orthogonal)
					proj = (dot_gv / (v_norm_sq + 1e-8)) * velocity
					g_orthogonal = grad - proj

					# Apply Force: Effectively boosting the orthogonal component
					# New Grad = Parallel + (1 + lambda) * Orthogonal
					p.grad.add_(g_orthogonal, alpha=self.steering_strength)

				else:
					# Raw Proposal 4 (Triple Product Gauge)
					# Force = lambda * [ ||v||^2 g - (g . v) v ]
					# This scales with ||v||^2, so lambda must be tuned carefully.
					# This corresponds to Eq. 23 in the derivation.

					# Term 1: ||v||^2 * g
					term1 = v_norm_sq * grad
					# Term 2: (g . v) * v
					term2 = dot_gv * velocity

					force = self.steering_strength * (term1 - term2)

					# In physics, F_total = -grad + F_mag.
					# The optimizer update is:
					# theta -= lr * (-F_total) = lr * (grad - F_mag).
					# So we SUBTRACT the magnetic force from the gradient.
					p.grad.sub_(force)

	# Proxy other method calls to base optimizer
	def __getattr__(self, name):
		return getattr(self.base_optimizer, name)



class MaxwellSteeringOptimizer(torch.optim.Optimizer):
	def __init__(self, base_optimizer, steering_strength=1.0):
		self.base_optimizer = base_optimizer
		self.steering_strength = steering_strength

		super().__init__(
			self.base_optimizer.param_groups,
			self.base_optimizer.defaults,
		)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		# Apply steering BEFORE the update
		self._apply_maxwell_steering()

		# Perform the base update
		self.base_optimizer.step()

		return loss

	def zero_grad(self, set_to_none=True):
		self.base_optimizer.zero_grad(set_to_none=set_to_none)

	@torch.no_grad()
	def _apply_maxwell_steering(self):
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue

				# CRITICAL FIX:
				# Check if param exists in state without creating a default empty dict.
				# If we trigger creation, we break Adam's lazy init logic.
				if p not in self.state:
					continue

				state = self.state[p]

				# Double check: if state is empty, Adam hasn't run yet. Do nothing.
				if len(state) == 0:
					continue

				# Identify Velocity (Momentum)
				velocity = None
				if 'exp_avg' in state:  # Adam/AdamW
					velocity = state['exp_avg']
				elif 'momentum_buffer' in state:  # SGD
					velocity = state['momentum_buffer']

				# If no momentum exists (first step or no momentum opt), we cannot steer.
				if velocity is None:
					continue

				# --- Logic for Proposal 2 (Maxwell) ---

				# We need a 'prev_grad' buffer.
				# Since we are now guaranteed that state is initialized, we can safely add it.
				if 'prev_grad' not in state:
					state['prev_grad'] = torch.clone(p.grad).detach()
					# Cannot compute flux on the very first step of tracking
					continue

				# Calculate Flux (Change in Gradient)
				delta_g = p.grad - state['prev_grad']

				# Update buffer immediately for next step
				state['prev_grad'].copy_(p.grad)

				# Physics: Apply Orthogonal Steering
				v_flat = velocity.view(-1)
				d_flat = delta_g.view(-1)

				v_norm_sq = torch.dot(v_flat, v_flat)

				# Avoid division by zero
				if v_norm_sq < 1e-12:
					continue

				# Project Flux onto Velocity direction
				proj_coeff = torch.dot(d_flat, v_flat) / v_norm_sq
				proj = proj_coeff * velocity

				# Orthogonal component of the flux (Pure Steering Force)
				flux_steer = delta_g - proj

				# Apply the force.
				# Subtracting from p.grad effectively ADDS the force to the particle
				# (because update is theta = theta - lr * grad)
				p.grad.sub_(flux_steer, alpha=self.steering_strength)
