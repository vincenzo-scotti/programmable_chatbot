import warnings
import torch
import math

from typing import Union


class LinearAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    EPSILON = 1.e-9

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            steps: int = 1,
            last_epoch: int = -1,
            warmup: Union[int, float] = 0
    ):
        assert isinstance(warmup, int) or 0. < warmup <= 1.

        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr: float = optimizer.defaults['lr']
        self.steps: int = steps
        self.idx: int = 0
        self.warmup: int = warmup if isinstance(warmup, int) else math.ceil(warmup * self.lr_steps)
        self.lr_values = torch.cat([
            torch.linspace(self.EPSILON, self.lr, self.warmup),
            torch.linspace(self.lr, self.EPSILON, self.steps - self.warmup)
        ])

        super(LinearAnnealingLR, self).__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self):
        warnings.warn(torch.optim.lr_scheduler.SAVE_STATE_WARNING, UserWarning)
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer',)}

        return state_dict

    def load_state_dict(self, state_dict):
        warnings.warn(torch.optim.lr_scheduler.SAVE_STATE_WARNING, UserWarning)

        self.lr = state_dict.pop('lr')
        self.steps = state_dict.pop('steps')
        self.idx = state_dict.pop('idx')
        self.warmup = state_dict.pop('warmup')

        self.lr_values = torch.cat([
            torch.linspace(self.EPSILON, self.lr, self.warmup),
            torch.linspace(self.lr, self.EPSILON, self.steps - self.warmup)
        ])

        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")
        try:
            lr = self.lr_values[self.idx]
        except IndexError:
            lr = self.EPSILON

        self.idx += 1

        return [lr]
