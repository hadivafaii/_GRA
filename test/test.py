from utils.plotting import *
from dataclasses import dataclass
from main.dataset import make_dataloader


# -------------------------
# Model
# -------------------------

class ConvNet(torch.nn.Module):
    def __init__(
            self,
            num_ftrs: int = 32,
            activation=torch.nn.SiLU,
            num_classes: int = 10,
            dropout: float = 0.5,
    ):
        super().__init__()

        f = num_ftrs
        c1, c2, c3, c4, c5 = f, 2 * f, 4 * f, 4 * f, 8 * f
        act = lambda: _make_act(activation)

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, c1, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c1),
            act(),

            torch.nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c2),
            act(),

            torch.nn.MaxPool2d(2, 2),  # 16x16

            torch.nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c3),
            act(),

            torch.nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c4),
            act(),

            torch.nn.MaxPool2d(2, 2),  # 8x8

            torch.nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c5),
            act(),

            torch.nn.MaxPool2d(2, 2),  # 4x4
        )

        hidden = 16 * f
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(c5 * 4 * 4, hidden),
            act(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _make_act(act_cls):
    try:
        return act_cls(inplace=True)
    except TypeError:
        return act_cls()


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_print_every(epochs: int) -> int:
    if epochs <= 60:
        return 1
    if epochs <= 300:
        return 5
    if epochs <= 1000:
        return 10
    if epochs <= 5000:
        return 25
    return 50


# -------------------------
# Training
# -------------------------

@dataclass
class TrainConfig:
    dataset: str = "CIFAR10"
    epochs: int = 100
    device: Union[str, torch.device] = "cuda:0"
    print_every: Optional[int] = None
    grad_clip: Optional[float] = None
    amp: bool = False  # mixed precision
    verbose: bool = True


def make_opt(opt_class, **opt_kws):
    def _fn(params):
        return opt_class(params, **opt_kws)
    return _fn


@torch.no_grad()
def evaluate_accuracy(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        out = model(x)
        pred = out.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return 100.0 * correct / max(1, total)


def train_model(
        opt_fn,
        *,
        model_class=ConvNet,
        model_kws: Dict[str, Any] = None,
        cfg: TrainConfig = TrainConfig(),
        seed: Optional[int] = None, ) -> Dict[str, np.ndarray]:
    if not isinstance(cfg.device, torch.device):
        device = torch.device(cfg.device)
    else:
        device = cfg.device
    if seed is not None:
        set_seed(seed)

    model_kws = {} if model_kws is None else dict(model_kws)

    trn, vld = make_dataloader(cfg.dataset, device=device)
    model = model_class(**model_kws).to(device)

    optimizer = opt_fn(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.amp))

    if cfg.print_every is not None:
        print_every = cfg.print_every
    else:
        print_every = default_print_every(cfg.epochs)

    if cfg.verbose:
        lines = [
            f"\n--- Training ({cfg.dataset}) ---",
            f"\nModel: {model_class.__name__}"
            + (f"({model_kws})" if model_kws else "")
            + (f" | seed={seed}" if seed is not None else ""),
            f"Optimizer: {optimizer.__class__.__name__}",
        ]
        print("\n".join(lines))
        print(optimizer)

    train_loss = []
    test_acc = []

    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for x, y in trn:
            x, y = x.to(device), y.to(device).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()

            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        train_loss.append(avg_loss)

        acc = evaluate_accuracy(model, vld)
        test_acc.append(acc)

        if cfg.verbose:
            should_print = (
                (epoch == 0) or
                (epoch + 1) == cfg.epochs or
                ((epoch + 1) % print_every == 0)
            )
            if should_print:
                print('  '.join([
                    f"Epoch {epoch+1:>4}/{cfg.epochs}:",
                    f"Train Loss {avg_loss:.4f}",
                    f"|  Test Acc {acc:.2f}%",
                ]))

    if cfg.verbose:
        print(f"Finished in {time.time() - start_time:.1f}s")

    results = {
        "train_loss": np.asarray(train_loss, dtype=np.float64),
        "test_acc": np.asarray(test_acc, dtype=np.float64),
    }
    return results


# -------------------------
# Experiments
# -------------------------

@dataclass
class Experiment:
    name: str
    opt_fn: Callable


def run_experiments(
        experiments: Sequence[Experiment],
        *,
        cfg: TrainConfig,
        model_class=ConvNet,
        model_kws: Optional[Dict[str, Any]] = None,
        seeds: Optional[Sequence[int]] = None, ) -> Dict[str, Dict[str, np.ndarray]]:
    results: Dict[str, Dict[str, np.ndarray]] = {}

    if seeds is None:
        for exp in experiments:
            hist = train_model(
                exp.opt_fn,
                model_class=model_class,
                model_kws=model_kws,
                cfg=cfg,
                seed=None,
            )
            results[exp.name] = hist
        return results

    for exp in experiments:
        per_seed = []
        for s in seeds:
            hist = train_model(
                exp.opt_fn,
                model_class=model_class,
                model_kws=model_kws,
                cfg=cfg,
                seed=int(s),
            )
            per_seed.append(hist)

        # stack into shape (n_seeds, T)
        keys = per_seed[0].keys()
        stacked = {k: np.stack([
            h[k] for h in per_seed
        ], axis=0) for k in keys}
        results[exp.name] = stacked

    return results


# -------------------------
# Plotting
# -------------------------

def _smooth_1d(y: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return y
    w = np.ones(int(k), dtype=np.float64) / float(k)
    return np.convolve(y, w, mode="same")


def plot_results(
        results: Dict[str, Dict[str, np.ndarray]],
        *,
        intvl: Optional[slice] = None,
        log_scale: bool = False,
        figsize: tuple = (12, 5),
        dpi : float = 100,
        smooth: int = None,
        title: str = "", ):
    if intvl is None:
        intvl = slice(0, None)

    fig, (ax_loss, ax_acc) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi)

    for name, d in results.items():
        loss = d["train_loss"]
        acc = d["test_acc"]

        # single run: (T,)
        if loss.ndim == 1:
            y = _smooth_1d(loss[intvl], smooth)
            ax_loss.plot(y, label=name, alpha=0.85)
        # multi-seed: (S, T)
        else:
            y = np.stack([_smooth_1d(v[intvl], smooth) for v in loss], axis=0)
            mu, sd = y.mean(axis=0), y.std(axis=0)
            ax_loss.plot(mu, label=name, alpha=0.9)
            ax_loss.fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=0.15)

        if acc.ndim == 1:
            y = _smooth_1d(acc[intvl], smooth)
            ax_acc.plot(y, label=name, lw=1.0, marker="o", markersize=3)
        else:
            y = np.stack([_smooth_1d(v[intvl], smooth) for v in acc], axis=0)
            mu, sd = y.mean(axis=0), y.std(axis=0)
            ax_acc.plot(mu, label=name, lw=1.0, marker="o", markersize=3)
            ax_acc.fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=0.15)

    if log_scale:
        ax_loss.set_xscale('log')
    ax_loss.set_title("Training Loss" + (f" | {title}" if title else ""))
    ax_loss.set_xlabel("Epochs")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.set_title("Test Accuracy" + (f" | {title}" if title else ""))
    ax_acc.set_xlabel("Epochs")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()
    plt.show()
