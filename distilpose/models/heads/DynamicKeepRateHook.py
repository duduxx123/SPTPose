from mmcv.runner import Hook, HOOKS
import math

@HOOKS.register_module()
class DynamicKeepRateHook(Hook):
    def __init__(self, warmup_epochs=10, total_decay_epochs=100, base_keep_rate=0.7, max_keep_rate=1.0):
        self.warmup_epochs = warmup_epochs
        self.total_decay_epochs = total_decay_epochs
        self.base_keep_rate = base_keep_rate
        self.max_keep_rate = max_keep_rate

    def adjust_keep_rate(self, iters, epoch, iter_per_epoch):
        if epoch < self.warmup_epochs:
            return 1.0
        if epoch >= self.total_decay_epochs:
            return self.base_keep_rate
        total_decay_iters = iter_per_epoch * (self.total_decay_epochs - self.warmup_epochs)
        iters = iters - iter_per_epoch * self.warmup_epochs
        keep_rate = self.base_keep_rate + (self.max_keep_rate - self.base_keep_rate) * \
            (math.cos(iters / total_decay_iters * math.pi) + 1) * 0.5
        return keep_rate

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        cur_epoch = runner.epoch
        iter_per_epoch = len(runner.data_loader)
        new_keep_rate = self.adjust_keep_rate(cur_iter, cur_epoch, iter_per_epoch)
        if hasattr(runner.model, 'update_keep_rate'):
            runner.model.update_keep_rate(new_keep_rate)
        else:
            for m in runner.model.modules():
                if hasattr(m, 'keep_rate'):
                    m.keep_rate = new_keep_rate
        runner.logger.info(f"Epoch {cur_epoch}, Iter {cur_iter}: Updated keep_rate to {new_keep_rate:.4f}")
