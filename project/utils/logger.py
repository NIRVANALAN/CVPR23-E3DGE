try:
    import wandb
except ImportError:
    wandb = None


def log_metrics(self, metrics_dict, prefix):
    for key, value in metrics_dict.items():
        self.logger.add_scalar('{}/{}'.format(prefix, key), value,
                               self.global_step)


def print_metrics(global_step, metrics_dict, prefix):
    print('Metrics for {}, step {}'.format(prefix, global_step))
    for key, value in metrics_dict.items():
        print('\t{} = '.format(key), value)
