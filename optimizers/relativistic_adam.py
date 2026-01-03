import torch


class RelativisticAdam(torch.optim.Optimizer):
    """
    Implements Relativistic Adam (RelAdam).

    A physics-derived optimizer modeling a relativistic particle in a viscoelastic
    spacetime. It smoothly interpolates between Newtonian mechanics (AdamW)
    and ultra-relativistic mechanics (clipped updates) via a single coupling.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate.
        betas (Tuple[float, float]): (beta1, beta2) coefficients for moment EMAs.
        eps (float): Term added to the denominator for numerical stability.
        weight_decay (float): Decoupled weight decay coefficient (AdamW-style).
        coupling (float): Dimensionless coupling C.
            - C -> Infinity : Newtonian limit (matches Adam/AdamW behavior).
            - C -> 0        : Strong relativistic suppression.
            - C = 3.0       : Standard "Safety Net" (clips 3-sigma outliers).
        inertial_sourcing (bool):
            - True: metric tracks momentum variance (p^2). More robust.
            - False: metric tracks gradient variance (g^2). Matches Adam sourcing.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        coupling=3.0,
        inertial_sourcing=False,
    ):
        coupling = _normalize_coupling(coupling)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if coupling < 0.0:
            raise ValueError(f"Invalid coupling: {coupling}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            coupling=coupling,
            inertial_sourcing=inertial_sourcing,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            coupling = group["coupling"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RelativisticAdam" \
                    "does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # 1) AdamW-style decoupled weight decay
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # 2) First moment (Momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3) Lorentz Factor (Gamma)
                # At step 1, variance is 0. Force Newtonian start.
                if coupling == float("inf") or step == 1:
                    gamma = 1.0
                else:
                    # We use the previous step's variance estimate for stability (Explicit Euler)
                    metric_psi = exp_avg_sq.sqrt().add(eps)
                    p_scaled = exp_avg / metric_psi
                    
                    # Gamma = sqrt(1 + (Z/C)^2)
                    gamma = (1.0 + (p_scaled / coupling).pow(2)).sqrt()

                # 4) Second moment (Metric)
                # Switch: Inertial (p^2) vs Gradient (g^2) sourcing
                if group["inertial_sourcing"]:
                    source_proxy = exp_avg
                else:
                    source_proxy = grad

                source_sq = source_proxy.pow(2)

                # Relativistic Source Suppression: T ~ E / gamma
                if isinstance(gamma, float) and gamma == 1.0:
                    source_term = source_sq
                else:
                    source_term = source_sq / gamma

                exp_avg_sq.mul_(beta2).add_(source_term, alpha=1.0 - beta2)

                # 5) Bias Correction & Step
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                step_size = lr / bc1
                
                # Denominator = (sqrt(v_hat) * gamma) + eps
                # Note: We apply gamma to the RMS term, then add epsilon floor.
                v_hat = exp_avg_sq / bc2
                denom = v_hat.sqrt()
                
                if not (isinstance(gamma, float) and gamma == 1.0):
                    denom.mul_(gamma)
                
                denom.add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def _normalize_coupling(coupling):
    if coupling is None: return float("inf")
    if isinstance(coupling, str):
        s = coupling.strip().lower()
        if s in {"inf", "+inf", "infty", "infinite", "infinity", "∞", "+∞"}:
            return float("inf")
        return float(s)
    return float(coupling)


class PhysicalAdam(torch.optim.Optimizer):
    """
    Implements the 'Physical Adam' algorithm derived from Relativistic Dilaton Gravity.
    
    The key difference from standard Adam is the metric update:
    - Standard Adam: v_t = EMA(g_t^2)  (Metric tracks raw gradient variance)
    - Physical Adam: v_t = EMA(m_t^2)  (Metric tracks momentum kinetic energy)
    
    This results in an optimizer that normalizes steps based on the 'smooth' 
    momentum magnitude rather than instantaneous gradient noise.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PhysicalAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. Gradient (Force)
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (Momentum)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared momentum values (Metric Variance)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 2. Update Momentum (The Learner's Velocity)
                # p_t = beta1 * p_{t-1} + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. Update Metric (The Gravity Response)
                # CRITICAL CHANGE: The source is Momentum Squared (exp_avg^2), not Grad Squared!
                # v_t = beta2 * v_{t-1} + (1 - beta2) * (p_t)^2
                momentum_sq = exp_avg.pow(2)
                exp_avg_sq.mul_(beta2).add_(momentum_sq, alpha=1 - beta2)

                # 4. Bias Correction (Standard Adam logic to handle initialization zeros)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2))).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # 5. The Step
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
