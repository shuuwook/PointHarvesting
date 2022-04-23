class ProgressiveSampling:
    def __init__(self, epochs_limit, step_num, start_samples, end_samples):
        self.epochs_limit = 0 if epochs_limit == 0 \
                            else epochs_limit - epochs_limit % (epochs_limit // step_num)
        self.step_num = step_num
        self.start_samples = start_samples
        self.end_samples = end_samples

        self.interval = int((end_samples - start_samples)/(step_num-1))
        self.step = int(epochs_limit / step_num)

    def progress_update(self, epoch):
        if epoch < self.epochs_limit:
            self.sample_num = self.start_samples + (self.interval * (epoch // self.step))
        else:
            self.sample_num = self.end_samples

    def get_sample_num(self):
        return self.sample_num
