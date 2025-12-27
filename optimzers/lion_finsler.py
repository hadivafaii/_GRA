import math
import torch


class FinslerLion(torch.optim.Optimizer):
    """
    FinslerLion Optimizer.

    This is a Finsler-geometry generalization of Relativistic Lion where the
    speed limit is defined by an ℓ_p norm on velocity. The induced “massive”
    relativistic saturation depends on the dual norm ℓ_q of the gauge-shifted
    signal c_t, with 1/p + 1/q = 1.

    Core update (per parameter tensor):
      c_t = β1 m_{t-1} + (1-β1) g_t

      s_t = ||c_t||_q
      d_t = sign(c_t) * |c_t|^{q-1} / ||c_t||_q^{q-1}          (for finite p)
            sign(c_t)                                        (for p = ∞, q = 1)

      α_t = s_t / sqrt(s_t^2 + ρ^2),   where ρ = mass * lr * (1-β2)

      v_t = α_t * d_t

      θ <- θ - lr * v_t

    Special cases:
      - p = ∞, mass = 0: v_t = sign(c_t) (standard Lion)
      - p = ∞, mass > 0: v_t = (||c_t||_1 / sqrt(||c_t||_1^2 + ρ^2)) * sign(c_t)
                         (global scaling of sign)
      - finite p, mass = 0: v_t = d_t (massless saturated direction under ℓ_p)
      - finite p, mass > 0: smooth saturation via α_t with s_t = ||c_t||_q

    Notes:
      - This implementation computes norms per parameter tensor (not across all parameters).
      - Curvature correction (if enabled) matches the sign convention in your baseline code:
            g̃_t = g_t + kappa * (g_t - g_{t-1})
        Set use_curvature=False to disable.

    Args:
        params: model parameters
        lr: learning rate
        betas: (beta1, beta2)
        weight_decay: decoupled weight decay
        mass: >= 0, controls saturation smoothness via ρ = mass * lr * (1-beta2)
        p: Finsler primal norm p in (1, ∞] (use math.inf / np.inf / "inf"). Determines dual q.
        use_curvature: curvature correction flag
        eps: numerical stability for normalization
    """

    @staticmethod
    def _normalize_p(p):
        if p is None:
            raise ValueError("p must be a number or inf")

        # Strings like "inf", "np.inf", "math.inf", "infinity", "∞"
        if isinstance(p, str):
            s = p.strip().lower().replace(" ", "")
            if s in {
                "inf", "+inf", "infty", "+infty", "infinity", "+infinity", "∞",
                "np.inf", "numpy.inf", "math.inf"
            }:
                return math.inf
            if s in {"-inf", "-infty", "-infinity"}:
                raise ValueError("Invalid p: -inf (use p > 1 or inf)")
            try:
                p = float(s)
            except ValueError as e:
                raise ValueError(f"Invalid p string: {p!r}") from e

        # Numbers (including numpy scalars and 0-d torch tensors)
        try:
            p_val = float(p)
        except (TypeError, ValueError) as e:
            raise ValueError(f"p must be a number or inf, got {type(p)}") from e

        if math.isnan(p_val):
            raise ValueError("Invalid p: NaN")
        if math.isinf(p_val):
            if p_val < 0:
                raise ValueError("Invalid p: -inf (use p > 1 or inf)")
            return math.inf

        return p_val

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        mass=0.0,
        p=math.inf,
        use_curvature=False,
        eps=1e-12,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0):
            raise ValueError(f"Invalid beta1: {beta1} (must be in (0,1))")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2} (must be in [0,1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} (must be >= 0)")
        if mass < 0.0:
            raise ValueError(f"Invalid mass: {mass} (must be >= 0)")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps} (must be > 0)")

        p = self._normalize_p(p)

        # Allow p = inf, or p > 1 for a well-behaved dual exponent q.
        if (not math.isinf(p)) and not (p > 1.0):
            raise ValueError(f"Invalid p: {p} (use p > 1 or inf)")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            mass=mass,
            p=p,
            kappa=(1.0 - beta1) / beta1,
            use_curvature=use_curvature,
            eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _dual_exponent(p: float) -> float:
        # For p = inf, q = 1. For finite p > 1, q = p/(p-1).
        if math.isinf(p):
            return 1.0
        return p / (p - 1.0)

    @staticmethod
    def _vector_norm(x: torch.Tensor, ord_val: float) -> torch.Tensor:
        # Compute ||x||_ord over all entries (flattened), returning a scalar tensor.
        if ord_val == math.inf:
            return x.abs().amax()
        if ord_val == 1.0:
            return x.abs().sum()
        return torch.linalg.vector_norm(x.reshape(-1), ord=ord_val)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            mass = group["mass"]
            p_norm = group["p"]
            q_norm = self._dual_exponent(p_norm)
            kappa = group["kappa"]
            use_curvature = group["use_curvature"]
            eps = group["eps"]

            # Effective mass scale (Δt=1):
            # ρ = m * c / τ = m * lr * (1 - beta2)
            rho = mass * lr * (1.0 - beta2)

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if use_curvature:
                        state["prev_grad"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                exp_avg = state["exp_avg"]

                # 1) Decoupled weight decay
                if wd > 0.0:
                    param.mul_(1.0 - lr * wd)

                # 2) Interpolation signal: c_t = β1 m_{t-1} + (1-β1) g_t
                c_t = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)

                # 3) Finsler-relativistic velocity update v_t
                if torch.count_nonzero(c_t) == 0:
                    update = torch.zeros_like(c_t)
                else:
                    if math.isinf(p_norm):
                        # q = 1, direction is sign(c_t)
                        d_t = torch.sign(c_t)
                        if mass == 0.0:
                            update = d_t
                        else:
                            s = self._vector_norm(c_t, ord_val=1.0)  # ||c||_1
                            alpha = s / torch.sqrt(s * s + (rho * rho))
                            update = alpha * d_t
                    else:
                        # finite p: q = p/(p-1)
                        s = self._vector_norm(c_t, ord_val=q_norm)  # ||c||_q
                        if s.item() == 0.0:
                            update = torch.zeros_like(c_t)
                        else:
                            abs_c = c_t.abs()
                            power = q_norm - 1.0
                            d_t = (
                                torch.sign(c_t)
                                * (abs_c.clamp_min(eps) ** power)
                                / (s.clamp_min(eps) ** power)
                            )
                            if mass == 0.0:
                                update = d_t
                            else:
                                alpha = s / torch.sqrt(s * s + (rho * rho))
                                update = alpha * d_t

                # 4) Parameter update
                param.add_(update, alpha=-lr)

                # 5) Curvature correction (optional)
                if use_curvature and state["step"] > 1:
                    prev_grad = state["prev_grad"]
                    grad_delta = grad - prev_grad
                    momentum_input = grad + kappa * grad_delta
                else:
                    momentum_input = grad

                # 6) Momentum update: m_t = β2 m_{t-1} + (1-β2) g̃_t
                exp_avg.mul_(beta2).add_(momentum_input, alpha=1.0 - beta2)

                # 7) Store gradient
                if use_curvature:
                    state["prev_grad"].copy_(grad)

        return loss
