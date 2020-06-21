from torch.optim.lr_scheduler import LambdaLR, CyclicLR


class LinearLR(LambdaLR):
    """
        A linear learning rate scheduler with a warmup.

        Keeps the lr at its initial value during the warmup
        period, then decreases it during the intermediate period,
        and finally keeps at the minimum level for the remaining time.
    """
    def __init__(self, optimizer, warmup, intermediate, no_bs_per_epoch,
                 lr_max, lr_min):
        """
            optimizer - the optimizer object.
            warmup - int; number of warmup iterations.
            intermediate - int; number of intermediate iterations.
            no_bs_per_epoch - number of batches in a single epoch.
            lr_max, lr_min - float; during the intermediate stage,
            the scheduler decreases the lr from lr_max to lr_min value.
        """
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
    """
        A slanted triangular cyclic learning rate scheduler with
        an exponentially decaying amplitude.
    """
    def __init__(self, optimizer, base_lr, max_lr, steps_per_cycle,
                 fraction_steps_up, gamma_factor,
                 cycle_momentum=False, **kwargs):
        """
            Args:
                optimizer - the optimizer object.
                base_lr, max_lr - float; the lower and upper bounds
                of the learning rate cycle.
                steps_per_cycle - int; number of steps in one cycle.
                fraction_steps_up - float; fraction of 'steps_per_cycle'
                that are spent on increasing the lr.The rest is spent on
                decreasing it back.
                gamma_factor - a factor by which the max_lr is multiplied
                after each cycle.
        """
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
