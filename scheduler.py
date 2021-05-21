from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

"""
 This is the official PyTorch tutorial of optimizer and scheduler.
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
"""


class SchedulerFactory:
    """
    A scheduler factory lass to generate learning rate schedulers.
    >>> factory = SchedulerFactory(CONFIG, 'plateau', opt)
    >>> scheduler = factory()
    OR simply:
    >>> scheduler = SchedulerFactory(CONFIG, 'plateau', opt)()
    """

    def __init__(self, config, option, optimizer):
        self.conf = config
        self.EXP_DECAY = 0.95

        if option == "plateau":
            self._scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=5)
        elif option == "linear":
            self._scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.linear_step)
        elif option == "exp":
            self._scheduler = LambdaLR(optimizer=optimizer, lr_lambda=self.exp_step)
        else:
            raise NotImplementedError(
                "Other schedulers have not been implemented yet.\nPlease input: plateau/linear/exp"
            )

    def __call__(self, *args, **kwargs):
        return self._scheduler

    def linear_step(self, epoch):
        return 1.0 - max(
            0, epoch + self.conf.start_epoch - self.conf.start_decay_epoch
        ) / (self.conf.num_epoch - self.conf.start_decay_epoch)

    def exp_step(self, epoch):
        return max(1.0, self.EXP_DECAY ** (epoch - self.conf.start_decay_epoch))
