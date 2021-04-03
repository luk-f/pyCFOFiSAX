import time


class ProfilingContext:
    data = {}

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        ProfilingContext.data.setdefault(self.name, []).append(duration)

    @classmethod
    def print_summary(cls):
        for name, data in cls.data.items():
            print(' --- {}: {}s avg for {} calls'.format(name, sum(data)/len(data), len(data)))
        cls.data = {}
