import torch
from torch.utils.data import DataLoader

__all__ = ['get_sampler', 'generate_train_dataloaders_default', 'generate_trainers_default',
           'generate_trainer_default', 'generate_test_dataloader_default', 'generate_tester_default']


def get_sampler(cfg_sampler, dataset, default='sequential'):
    from torch.utils.data import SequentialSampler, RandomSampler

    assert default in ['sequential', 'random'], 'Please choose a default sampler from "sequential", "random".'
    sampler_dict = {x.__name__: x for x in [SequentialSampler, RandomSampler]}
    if cfg_sampler is None:
        if default == 'sequential':
            sampler_typ = SequentialSampler
        elif default == 'random':
            sampler_typ = RandomSampler
        else:
            raise NotImplementedError(f'Default sampler type "{default}" not supported.')
        params = dict()
    else:
        cfg_sampler = cfg_sampler.copy()
        typ = cfg_sampler.pop('type')
        sampler_typ = sampler_dict.get(typ)
        if sampler_typ is None:
            raise ValueError(f'Configured sampler type {sampler_typ} not supported.')
        params = cfg_sampler
    return sampler_typ(dataset, **params)


def generate_train_dataloaders_default(cfg, num_clients):
    train_set = cfg.train_set

    # divisions of train set
    params = train_set['params']
    if train_set['root'] is not None:
        params.update({'root': train_set['root']})
    params.update({'train_transform': train_set['transform']})
    wrapper = train_set['type'](**params)
    params = train_set['division']
    if params is None:
        print('Dataset division options not specified in configuration file. Use iid by default.')
        params = {'method': 'iid'}
    params['num_clients'] = num_clients
    divisions = wrapper.get_train_set_division(**params)

    batch_size = train_set.get('batch_size', 1)
    num_workers = cfg.args['num_workers']
    cfg_sampler = train_set.get('sampler')

    dataloaders = []
    for division in divisions.values():
        sampler = get_sampler(cfg_sampler, division, default='random')
        dataloader = DataLoader(division, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        dataloaders.append(dataloader)

    return dataloaders


def generate_trainers_default(cfg, dataloaders, trainer_type, extra_params=None):
    cfg_model = cfg.model
    cfg_loss = cfg.loss
    cfg_optimizer = cfg.optimizer
    cfg_lr_scheduler = cfg.lr_scheduler
    device = cfg.device
    verbose = cfg.args['verbose_train']

    # gradient accumulation
    global_batch_size = cfg.train_set.get('global_batch_size')
    grad_acc_num_iters = None
    if global_batch_size is not None:
        batch_size = cfg.train_set.get('batch_size', 1)
        assert isinstance(global_batch_size, int)
        assert global_batch_size > batch_size
        grad_acc_num_iters = global_batch_size // batch_size
        global_batch_size = grad_acc_num_iters * batch_size
        print(f'Using gradient accumulation: global_batch_size={global_batch_size}')

    public_model = cfg.args['public_model'] > 0
    if public_model:
        model = cfg_model['type'](**cfg_model['params'])
        criterion = cfg_loss['type'](**cfg_loss['params'])
        params = cfg_optimizer['params']
        params['params'] = model.parameters()
        optimizer = cfg_optimizer['type'](**params)
        if cfg_lr_scheduler is not None:
            params = cfg_lr_scheduler['params']
            params['optimizer'] = optimizer
            lr_scheduler = cfg_lr_scheduler['type'](**params)
        else:
            lr_scheduler = None

        trainers = []
        for i, dataloader in enumerate(dataloaders):
            params = {
                'dataloader': dataloader,
                'model': model,
                'criterion': criterion,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'device': device,
                'verbose': verbose,
                'public_model': public_model,
                'grad_acc_num_iters': grad_acc_num_iters,
            }
            if extra_params is not None:
                params.update(extra_params)
            trainer = trainer_type(**params)
            trainers.append(trainer)
    else:
        trainers = []
        for i, dataloader in enumerate(dataloaders):
            model = cfg_model['type'](**cfg_model['params'])
            criterion = cfg_loss['type'](**cfg_loss['params'])
            params = cfg_optimizer['params']
            params['params'] = model.parameters()
            optimizer = cfg_optimizer['type'](**params)
            if cfg_lr_scheduler is not None:
                params = cfg_lr_scheduler['params']
                params['optimizer'] = optimizer
                lr_scheduler = cfg_lr_scheduler['type'](**params)
            else:
                lr_scheduler = None

            params = {
                'dataloader': dataloader,
                'model': model,
                'criterion': criterion,
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'device': device,
                'verbose': verbose,
                'public_model': public_model,
                'grad_acc_num_iters': grad_acc_num_iters,
            }
            if extra_params is not None:
                params.update(extra_params)
            trainer = trainer_type(**params)
            trainers.append(trainer)

    return trainers


def generate_trainer_default(cfg, dataloader, trainer_type, extra_params=None):
    """
    Applicable for centralized training (single client) only.
    """
    cfg_model = cfg.model
    cfg_loss = cfg.loss
    cfg_optimizer = cfg.optimizer
    cfg_lr_scheduler = cfg.lr_scheduler
    device = cfg.device
    verbose = cfg.args['verbose_train']

    # gradient accumulation
    global_batch_size = cfg.train_set.get('global_batch_size')
    grad_acc_num_iters = None
    if global_batch_size is not None:
        batch_size = cfg.train_set.get('batch_size', 1)
        assert isinstance(global_batch_size, int)
        assert global_batch_size > batch_size
        grad_acc_num_iters = global_batch_size // batch_size
        global_batch_size = grad_acc_num_iters * batch_size
        print(f'Using gradient accumulation: global_batch_size={global_batch_size}')

    # load model (if pretrained model is specified, load it because we want the client to use it)
    model = cfg_model['type'](**cfg_model['params'])
    pretrained_path = cfg_model['pretrained']
    if pretrained_path is not None:
        print(f'Loading pretrained model {pretrained_path}...')
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

    criterion = cfg_loss['type'](**cfg_loss['params'])

    params = cfg_optimizer['params']
    params['params'] = model.parameters()
    optimizer = cfg_optimizer['type'](**params)

    if cfg_lr_scheduler is not None:
        params = cfg_lr_scheduler['params']
        params['optimizer'] = optimizer
        lr_scheduler = cfg_lr_scheduler['type'](**params)
    else:
        lr_scheduler = None

    params = {
        'dataloader': dataloader,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'verbose': verbose,
        'public_model': False,
        'grad_acc_num_iters': grad_acc_num_iters,
    }
    if extra_params is not None:
        params.update(extra_params)
    trainer = trainer_type(**params)

    return trainer


def generate_test_dataloader_default(cfg):
    test_set = cfg.test_set

    params = test_set['params']
    if test_set['root'] is not None:
        params.update({'root': test_set['root']})
    params.update({'test_transform': test_set['transform']})
    wrapper = test_set['type'](**params)
    dataset = wrapper.get_test_set()
    cfg_sampler = test_set.get('sampler')
    sampler = get_sampler(cfg_sampler, dataset)

    batch_size = test_set.get('batch_size', 1)
    num_workers = cfg.args['num_workers']

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return dataloader


def generate_tester_default(cfg, dataloader):
    cfg_model = cfg.model
    cfg_tester = cfg.tester
    tester_type = cfg_tester['type']
    device = cfg.device
    verbose = cfg.args['verbose_val']

    pretrained_path = cfg_model['pretrained']
    model = cfg_model['type'](**cfg_model['params'])
    if pretrained_path is not None:
        print(f'Loading pretrained model {pretrained_path}...')
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

    tester = tester_type(dataloader, model, device, verbose,
                         **cfg_tester['params'])

    return tester
