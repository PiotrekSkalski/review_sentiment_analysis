from torch.optim.lr_scheduler import LambdaLR, CyclicLR


class LinearLR(LambdaLR):
    def __init__(self, optimizer, warmup, intermediate, no_bs_per_epoch,
                 lr_max, lr_min):
        self.warmup = warmup*no_bs_per_epoch
        self.intermediate = intermediate*no_bs_per_epoch
        self.no_bs_per_epoch = no_bs_per_epoch
        self.lr_min = lr_min
        self.lr_max = lr_max
        super().__init__(optimizer, self.stepper)

    def stepper(self, epoch):
        if self._step_count <= self.warmup:
            return 1.0
        elif self._step_count > (self.warmup + self.intermediate):
            return self.lr_min/self.lr_max
        else:
            return (1 - (self._step_count - self.warmup)
                    * (1 - self.lr_min/self.lr_max)/self.intermediate)


class CyclicLRDecay(CyclicLR):
    def __init__(self, optimizer, base_lr, max_lr, steps_per_cycle,
                 fraction_steps_up, gamma_factor,
                 cycle_momentum=False, **kwargs):
        self.gamma_factor = gamma_factor
        step_size_up = int(steps_per_cycle * fraction_steps_up)
        step_size_down = steps_per_cycle - step_size_up
        super().__init__(optimizer, base_lr, max_lr, step_size_up,
                         step_size_down, cycle_momentum=cycle_momentum,
                         **kwargs)

    def step(self, epoch=None):
        if not self._step_count % self.total_size and self._step_count != 0:
            self.max_lrs = [lr*self.gamma_factor for lr in self.max_lrs]
        super().step(epoch)
