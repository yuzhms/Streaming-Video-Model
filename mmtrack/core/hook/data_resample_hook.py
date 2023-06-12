from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DataResampleHook(Hook):
    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            data_loaders = runner.data_loader
            if not isinstance(data_loaders, (list, tuple)):
                data_loaders = [data_loaders]
            for data_loader in data_loaders:
                dataset = data_loader.dataset
                self._resample_dataset(runner, dataset)

    def _resample_dataset(self, runner, dataset):
        if hasattr(dataset, 'resample'):
            runner.logger.info(f'resample dataset {dataset.__class__.__name__}: {dataset}')
            dataset.resample()
        if hasattr(dataset, 'dataset'):
            self._resample_dataset(runner, dataset.dataset)
        if hasattr(dataset, 'datasets'):
            for _dataset in dataset.datasets:
                self._resample_dataset(runner, _dataset)