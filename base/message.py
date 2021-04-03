class Message:
    def __init__(self, metrics=None):
        self.metrics = metrics
        self.max_len = 0

    def msg_out(self, loss, accuracy=.0, is_wrap=False, **kwargs):
        if is_wrap:
            prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
            self.msg_println(self.make_msg(loss, accuracy, prefix))
        else:
            cur_batch = kwargs['cur_batch'] if 'cur_batch' in kwargs else None
            count_batch = kwargs['count_batch'] if 'count_batch' in kwargs else None
            self.msg_print(self.make_cover(loss, accuracy, cur_batch, count_batch))

    def make_cover(self, loss, accuracy=.0, cur_batch=None, count_batch=None):
        schedule = self.get_schedule(1. * cur_batch / count_batch)
        msg = self.make_msg(loss, accuracy, prefix='')
        return f'{cur_batch}/{count_batch} {schedule}{msg}'

    def make_msg(self, loss, accuracy=.0, prefix=''):
        msg = f' - {prefix}loss: {loss:.6f}'
        if self.metrics:
            msg += f' - {prefix}{self.metrics[0]}: {accuracy:.6f}'
        return msg

    @staticmethod
    def get_schedule(scale=1., count=25):
        schedule = f'[{"-" * count}]'
        schedule = schedule.replace('-', '=', int(scale * count))
        return schedule.replace('-', '>', 1)

    @staticmethod
    def msg_print(*arg, **kwargs):
        # 不换行, 覆盖
        if 'flush' in kwargs:
            kwargs.pop('flush')
        if 'end' in kwargs:
            kwargs.pop('end')
        print('\r', *arg, **kwargs, end='', flush=True)

    @staticmethod
    def msg_println(*arg, **kwargs):
        # 换行, 添加
        print(*arg, **kwargs)

    def set_metrics(self, metrics):
        self.metrics = metrics
