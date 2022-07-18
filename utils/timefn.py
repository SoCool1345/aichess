import threading
import time
import torch.multiprocessing
from functools import wraps


def timefn(fn):
    """计算性能的修饰器"""
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"process: {torch.multiprocessing.current_process().name} thread: {threading.current_thread().getName()} @timefn: {fn.__name__} took {t2 - t1: .9f} s")
        return result
    return measure_time
