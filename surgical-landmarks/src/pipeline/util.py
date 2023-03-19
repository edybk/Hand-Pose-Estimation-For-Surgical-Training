import time

class Timer():
    def __init__(self, name="block", verbose=False):
        self.verbose = verbose
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, type, value, traceback):
        if self.verbose:
            print(f"total time to run {self.name} is {time.time()-self.start_time}")