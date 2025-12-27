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


def load_model(
		s: str,
		device: str,
		lite: bool = False,
		**kwargs, ):
	kwargs.setdefault('device', device)
	if lite:
		tr, meta = load_model_lite(s, **kwargs)
	else:
		tr, meta = load_model_main(*s.split('/'), **kwargs)
	return tr, meta


def load_model_lite(
		path: str,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		**kwargs, ):
	# load model
	cfg = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' not in e
	)
	fname = cfg.split('.')[0]
	cfg = pjoin(path, cfg)
	with open(cfg, 'r') as f:
		cfg = json.load(f)
	# extract key
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](save=False, **cfg)
	from main.model import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# load state dict
	fname_pt = next(
		f for f in os.listdir(path)
		if f.split('.')[-1] == 'pt'
	)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(
		f=state_dict,
		map_location='cpu',
		weights_only=False,  # TODO: later revert to True
	)
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	cfg_train = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' in e
	)
	fname = cfg_train.split('.')[0]
	cfg_train = pjoin(path, cfg_train)
	with open(cfg_train, 'r') as f:
		cfg_train = json.load(f)
	if fname.endswith('Train'):
		from main.train import Trainer
		cfg_train = ConfigTrain(**cfg_train)
		kwargs.setdefault('shuffle', False)
		trainer = Trainer(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	# optim, etc.
	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	# stats
	stats_model = state_dict['metadata'].pop('stats_model', {})
	stats_trainer = state_dict['metadata'].pop('stats_trainer', {})
	trainer.model.stats.update(stats_model)
	trainer.stats.update(stats_trainer)
	# meta
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


def load_model_main(
		model_name: str,
		fit_name: Union[str, int] = -1,
		checkpoint: int = -1,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		path: str = 'Projects/TemporalSC/models',
		**kwargs, ):
	# cfg model
	path = pjoin(add_home(path), model_name)
	fname = next(s for s in os.listdir(path) if 'json' in s)
	with open(pjoin(path, fname), 'r') as f:
		cfg = json.load(f)
	# extract key
	fname = fname.split('.')[0]
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](save=False, **cfg)
	from main.model import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# now enter the fit folder
	if isinstance(fit_name, str):
		path = pjoin(path, fit_name)
	elif isinstance(fit_name, int):
		path = sorted(filter(
			os.path.isdir, [
				pjoin(path, e) for e
				in os.listdir(path)
			]
		), key=_sort_fn)[fit_name]
	else:
		raise ValueError(fit_name)
	files = sorted(os.listdir(path))

	# load state dict
	fname_pt = [
		f for f in files if
		f.split('.')[-1] == 'pt'
	]
	if checkpoint == -1:
		fname_pt = fname_pt[-1]
	else:
		fname_pt = next(
			f for f in fname_pt if
			checkpoint == _chkpt(f)
		)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(
		f=state_dict,
		map_location='cpu',
		weights_only=False,  # TODO: later revert to True
	)
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	fname = next(
		f for f in files if
		f.split('.')[-1] == 'json'
	)
	with open(pjoin(path, fname), 'r') as f:
		cfg_train = json.load(f)
	fname = fname.split('.')[0]
	if fname == 'ConfigTrain':
		from main.train import Trainer
		cfg_train = ConfigTrain(**cfg_train)
		kwargs.setdefault('shuffle', False)
		trainer = Trainer(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError(fname)

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	# stats
	stats_model = state_dict['metadata'].pop('stats_model', {})
	stats_trainer = state_dict['metadata'].pop('stats_trainer', {})
	trainer.model.stats.update(stats_model)
	trainer.stats.update(stats_trainer)
	# meta
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


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
