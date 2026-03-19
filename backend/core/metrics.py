class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0

    def record(self, latency):
        self.total_requests += 1
        self.total_latency += latency

    def error(self):
        self.total_errors += 1

    @property
    def avg_latency(self):
        if self.total_requests == 0:
            return 0
        return round(self.total_latency / self.total_requests, 4)

metrics = Metrics()