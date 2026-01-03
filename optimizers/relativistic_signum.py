import torch


class RelativisticSignum(torch.optim.Optimizer):
    """
    Relativistic Sign Momentum.

    Represents the exact solution to the relativistic equations of motion
    in the limit of zero gauge coupling (k=0). unlike Lion, there is no
    interpolation step; the velocity is determined solely by the
    accumulated kinematic momentum.

    Args:
        params: Model parameters
        lr: Learning rate η (physical interpretation: η = c·Δt)
        momentum: Momentum decay factor β (corresponds to Lion's β₂)
        weight_decay: Decoupled weight decay coefficient
        mass: Rest mass of the particle.
            mass=0: Exact Sign-Momentum (SignSGD with momentum)
            mass>0: Relativistic Sign-Momentum (Smooth saturation)

    Physical correspondence:
        - This is the β₁ = 1.0 limit of Relativistic Lion.
        - The velocity is determined by p_kinematic alone.
        - No curvature correction is applied (it vanishes exactly when k=0).
    """

    def __init__(
        self, params, lr=1e-3, momentum=0.9,
        weight_decay=0.0, mass=0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if mass < 0:
            raise ValueError(f"Invalid mass: {mass}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            mass=mass,
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
            beta = group['momentum']
            wd = group['weight_decay']
            mass = group['mass']
            
            # ρ = m·c/τ. 
            # Consistent with Lion derivation:
            # τ ≈ Δt/(1-β), c = η/Δt
            rho = mass * lr * (1 - beta)
            # TODO: verify

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize momentum buffer to 0 (or first gradient?)
                    # Standard practice is 0.
                    state['momentum_buffer'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                state['step'] += 1
                buf = state['momentum_buffer']

                # 1. Decoupled Weight Decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # 2. Momentum Update (Kinematic Accumulation)
                # m_t = β·m_{t-1} + (1-β)·g_t
                buf.mul_(beta).add_(grad, alpha=1 - beta)

                # 3. Relativistic Velocity: v = m_t / √(m_t² + ρ²)
                # In the massless limit (ρ=0), this becomes sign(m_t)
                if mass == 0:
                    update = torch.sign(buf)
                else:
                    # We create a tensor for rho to allow broadcasting if needed, 
                    # though scalar works for hypot in modern torch.
                    rho_t = buf.new_tensor(rho)
                    update = buf / torch.hypot(buf, rho_t)

                # 4. Parameter Update: θ_t = θ_{t-1} - η·v_t
                p.add_(update, alpha=-lr)

        return loss
