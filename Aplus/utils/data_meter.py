

class DataMeter:
    """Computes and stores current average and sum value"""
    def __init__(self):
        self.avg = None
        self.sum = None
        self.count = 0

    def reset(self):
        if self.avg is not None:
            self.avg *= 0
            self.sum *= 0
        self.count = 0

    def update(self, value, n_sample=1):
        if self.sum is None:
            self.sum = value * n_sample
        else:
            self.sum += value * n_sample
        self.count += n_sample
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum