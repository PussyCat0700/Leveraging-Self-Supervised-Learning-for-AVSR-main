from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """
    Subclassing torch.optim.lr_scheduler.ReduceLROnPlateau
    added warmup parameters
    """

    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-8,
                 warmup_itrs=40, warmup_type='lin', start_lr=1e-7, verbose=False):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, threshold_mode=threshold_mode,
                         cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)
        self.warmup_itrs = warmup_itrs  # 改了这个
        self.warmup_type = warmup_type
        self.start_lr = start_lr
        self.default_lrs = list()
        self.itr = 0
        # for param_group in optimizer.param_groups:
        #     self.default_lrs.append(param_group['lr'])
        self.default_lrs.append(1e-5)

    def step(self, metrics, epoch=None):
        if self.itr < self.warmup_itrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if self.warmup_type == 'exp':
                    new_lr = self.start_lr * (self.default_lrs[i] / self.start_lr) ** (self.itr / self.warmup_itrs)
                elif self.warmup_type == 'lin':
                    new_lr = self.start_lr + (self.default_lrs[i] - self.start_lr) * (self.itr / self.warmup_itrs)
                param_group['lr'] = new_lr
        elif self.itr == self.warmup_itrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.default_lrs[i]
        else:
            super().step(metrics)
        self.itr += 1
