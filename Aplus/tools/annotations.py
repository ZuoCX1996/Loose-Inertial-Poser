import time

def timing(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f'running {fn.__name__}')
        ret = fn(*args, **kwargs)
        end = time.time()
        print(f'done | time cost:{round(end-start, 3)} s')
        return ret
    return wrapper
