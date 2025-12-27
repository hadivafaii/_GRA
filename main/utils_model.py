# noinspection PyUnresolvedReferences
from utils.plotting import *
from torch import distributions as dists
import wandb


def compute_r2(true, pred):
	ss_res = (true - pred).pow(2)
	ss_tot = (true - true.mean(
		dim=(1, 2, 3), keepdim=True)).pow(2)
	ss_res = torch.sum(ss_res, dim=(1, 2, 3))
	ss_tot = torch.sum(ss_tot, dim=(1, 2, 3))
	# compute r2
	eps = torch.finfo(torch.float32).eps
	r2 = 1.0 - ss_res / (ss_tot + eps)
	return r2


def detach_fields(obj: Any, fields: List[str]):
	"""
	Detach obj.<field> if it exists and is a Tensor or a container
	(list/tuple) of Tensors. Leaves non-tensor fields untouched.
	"""
	for name in fields:
		if not hasattr(obj, name):
			continue
		val = getattr(obj, name)
		if val is None:
			continue
		if hasattr(val, 'detach'):
			setattr(obj, name, val.detach())
		elif isinstance(val, (list, tuple)):
			detached = []
			dirty = False
			for x in val:
				if hasattr(x, 'detach'):
					detached.append(x.detach())
					dirty = True
				else:
					detached.append(x)
			if dirty:
				setattr(obj, name, type(val)(detached))
	return


def print_grad_table(
		trainer,
		metadata: dict,
		clip: float = None,
		thresholds: List[float] = None, ):
	thresholds = thresholds if thresholds else [
		1, 2, 5, 10, 20, 50, 100, 200]
	clip = clip if clip else trainer.cfg.grad_clip
	thresholds = [
		clip * i for i
		in thresholds
	]
	bad = np.array(list(trainer.stats['grad'].values()))

	t = PrettyTable(['Threshold', '#', '%'])
	for thres in thresholds:
		tot = (bad > thres).sum()
		perc = tot / metadata['global_step']
		perc = np.round(100 * perc, 3)
		t.add_row([int(thres), tot, perc])
	print(t, '\n')
	return


def print_num_params(
		module: nn.Module,
		full: bool = True, ):
	def _tot_params(_m):
		return sum(
			p.numel() for p
			in _m.parameters()
			if p.requires_grad
		)

	def _add_module(name, _m):
		tot = _tot_params(_m)
		if tot == 0:
			return

		if tot >= 1e6:
			tot = f"{np.round(tot / 1e6, 2):1.1f} Mil"
		elif tot >= 1e3:
			tot = f"{np.round(tot / 1e3, 2):1.1f} K"
		else:
			tot = str(tot)
		t.add_row([name, tot])
		return

	def _process_module(_prefix, _m):
		if isinstance(_m, (nn.ModuleDict, nn.ModuleList)):
			for _name, sub_module in _m.named_children():
				full_name = f"{_prefix}.{_name}" \
					if _prefix else _name
				_process_module(full_name, sub_module)
		else:
			_add_module(_prefix, _m)

	# top row: the full module
	t = PrettyTable(['Module Name', 'Num Params'])
	_add_module(module.__class__.__name__, module)
	t.add_row(['———', '———'])

	for prefix, m in module.named_modules():
		if '.' in prefix or not prefix:
			continue
		if full:
			_process_module(prefix, m)
		else:
			_add_module(prefix, m)

	print(t, '\n')
	return


def add_weight_decay(
		model: nn.Module,
		weight_decay: float = 1e-2,
		skip: Tuple[str, ...] = ('bias',), ):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) <= 1 or any(k in name for k in skip):
			no_decay.append(param)
		else:
			decay.append(param)
	param_groups = [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay},
	]
	return param_groups


def track_code_state(
		wandb_run,
		project_dir: str,
		exclude_dirs: list = None,
		code_extensions: list = None, ):
	"""
	Track the state of the codebase and save it to wandb

	Args:
		wandb_run: The wandb run object
		project_dir:
		exclude_dirs: List of additional directories to exclude from
			tracking (will be added to default excluded directories)
		code_extensions: List of file extensions to track
			defaults to ['.py', '.sh'] if None
	"""
	default_exclude_dirs = {
		'__pycache__',
		'.git',
		'.ipynb_checkpoints',
		'.DS_Store',
		'wandb',
		'runs',
		'logs',
	}
	exclude_dirs = default_exclude_dirs.union(
		set(exclude_dirs or []))
	code_extensions = code_extensions or ['.py', '.sh']

	# Save git info
	try:
		import git
		repo = git.Repo(search_parent_directories=True)
		wandb_run.config.update({
			'git_commit': repo.head.object.hexsha,
			'git_branch': repo.active_branch.name if
				repo.active_branch else 'detached',
			'git_dirty': repo.is_dirty()
		})
	except Exception as e:
		print(f"Warning: Could not save code info: {str(e)}")

	# Save codebase
	for root, dirs, files in os.walk(project_dir):
		# Skip excluded directories
		dirs[:] = [d for d in dirs if d not in exclude_dirs]

		for file in files:
			if any(file.endswith(e) for e in code_extensions):
				file_path = os.path.join(root, file)
				wandb_run.save(file_path, base_path=str(project_dir))

	# Check if running in notebook
	try:
		import ipynbname
		nb_path = ipynbname.path()
		wandb_run.save(
			str(nb_path),
			base_path=str(nb_path.parent),
			policy='now',
		)
		wandb_run.config.update({
			'notebook_path': str(nb_path),
			'notebook_name': nb_path.name
		})
	except Exception as e:
		print(f"Warning: Could not save notebook info: {str(e)}")
	return


class DualLogger:
	def __init__(self, tb_writer, wandb_run):
		self.tb_writer = tb_writer
		self.wandb_run = wandb_run

	def add_scalar(self, tag, value, step):
		"""Write scalar to both TensorBoard and Wandb"""
		self.tb_writer.add_scalar(tag, value, step)
		self.wandb_run.log({tag: value}, step=step)

	def add_figure(self, tag, figure, step):
		"""Write figure to both TensorBoard and Wandb"""
		self.tb_writer.add_figure(tag, figure, step)
		self.wandb_run.log({tag: wandb.Image(figure)}, step=step)

	def close(self):
		"""Close both loggers"""
		if self.tb_writer:
			self.tb_writer.close()
		if self.wandb_run:
			self.wandb_run.finish()


class AverageMeter(object):
	def __init__(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def reset(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


class AverageMeterDict(object):
	def __init__(self, keys: Sequence[Any]):
		self.meters = {
			k: AverageMeter()
			for k in keys
		}

	def __getitem__(self, key):
		return self.meters.get(key)

	def __iter__(self):
		for key in self.meters:
			yield key

	def reset(self):
		for k, m in self.meters.items():
			m.reset()

	def update(self, vals: dict, n: int = 1):
		for k, m in self.meters.items():
			m.update(vals.get(k, 0), n)

	def avg(self):
		return {
			k: m.avg for k, m in
			self.meters.items()
		}


class Timer:
	def __init__(self):
		self.times = [time.time()]
		self.messages = {}
		self.counts = {}

	def __call__(self, message=None):
		self.times.append(time.time())
		t = self.times[-1] - self.times[-2]
		if message in self.messages:
			self.messages[message] += t
			self.counts[message] += 1
		else:
			self.messages[message] = t
			self.counts[message] = 1

	def print(self):
		for k, t in self.messages.items():
			print(f"{k}. Time: {t:0.1f}")


class Initializer:
	def __init__(self, dist_name: str, **kwargs):
		self.mode = 'pytorch'
		try:
			dist_module = getattr(dists, dist_name.lower())
			dist = getattr(dist_module, dist_name.title())
			kwargs = filter_kwargs(dist, kwargs)
		except AttributeError:
			dist = getattr(sp_stats, dist_name)
			self.mode = 'scipy'
		self.dist = dist(**kwargs)

	@torch.no_grad()
	def apply(self, weight: torch.Tensor):
		if self.mode == 'pytorch':
			values = self.dist.sample(weight.shape)
		else:
			values = self.dist.rvs(tuple(weight.shape))
			values = torch.tensor(values, dtype=torch.float)
		weight.data.copy_(values.to(weight.device))
		return


class Module(nn.Module):
	def __init__(self, cfg, verbose: bool = False):
		super(Module, self).__init__()
		self.chkpt_dir = None
		self.timestamp = now(True)
		self.stats = collections.defaultdict(list)
		self.verbose = verbose
		self.cfg = cfg

	def print(self):
		print_num_params(self)

	def create_chkpt_dir(self, fit_name: str = None):
		chkpt_dir = fit_name or '_'.join([
			f"seed-{self.cfg.seed}",
			f"({self.timestamp})",
		])
		chkpt_dir = pjoin(
			self.cfg.mods_dir,
			chkpt_dir,
		)
		os.makedirs(chkpt_dir, exist_ok=True)
		self.chkpt_dir = chkpt_dir
		return

	def save(
			self,
			checkpoint: int = -1,
			name: str = None,
			path: str = None, ):
		path = path if path else self.chkpt_dir
		name = name if name else type(self).__name__
		fname = '-'.join([
			name,
			f"{checkpoint:04d}",
			f"({now(True)}).pt",
		])
		fname = pjoin(path, fname)
		torch.save(self.state_dict(), fname)
		return fname


def _chkpt(f):
	return int(f.split('_')[0].split('-')[-1])


def _sort_fn(f: str):
	f = f.split('(')[-1].split(')')[0]
	ymd, hm = f.split(',')
	yy, mm, dd = ymd.split('_')
	h, m = hm.split(':')
	yy, mm, dd, h, m = map(
		lambda s: int(s),
		[yy, mm, dd, h, m],
	)
	x = (
		yy * 1e8 +
		mm * 1e6 +
		dd * 1e4 +
		h * 1e2 +
		m
	)
	return x
