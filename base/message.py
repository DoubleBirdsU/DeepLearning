class Message:
    def __init__(self, metrics=None):
        self.metrics = metrics
        self.msg_info = dict()
        self.max_len = 0

    def msg_out(self, loss=0., accuracy=.0, is_append=False, is_wrap=False, **kwargs):
        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        suffix = kwargs['suffix'] if 'suffix' in kwargs else ''
        msg_prefix = kwargs['msg_prefix'] if 'msg_prefix' in kwargs else ''
        if is_wrap:
            self.msg_println(msg_prefix, self.make_msg(loss, accuracy, prefix, suffix))
        elif is_append:
            cur_batch = kwargs['cur_batch'] if 'cur_batch' in kwargs else None
            count_batch = kwargs['count_batch'] if 'count_batch' in kwargs else None
            self.msg_print(msg_prefix, self.make_cover(loss, accuracy, cur_batch, count_batch))
        else:
            cur_batch = kwargs['cur_batch'] if 'cur_batch' in kwargs else None
            count_batch = kwargs['count_batch'] if 'count_batch' in kwargs else None
            self.msg_print(f'{msg_prefix} - Valid: {cur_batch}/{count_batch}')

    def make_cover(self, loss, accuracy=.0, cur_batch=None, count_batch=None):
        schedule = self.get_schedule(1. * cur_batch / count_batch)
        msg = self.make_msg(loss, accuracy, prefix='')
        return f'{cur_batch}/{count_batch} {schedule}{msg}'

    def make_msg(self, loss, accuracy=.0, prefix='', suffix=''):
        msg = f' - {prefix}loss: {loss:.6f}'
        if self.metrics:
            msg += f' - {prefix}{self.metrics[0]}: {accuracy:.6f}'
        return f'{msg}{suffix}'

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
