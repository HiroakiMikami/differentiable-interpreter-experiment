class Trigger:
    def __init__(self, interval: int, n_iter: int):
        self.interval = interval
        self.n_iter = n_iter

    def __call__(self, manager):
        return (manager.iteration == self.n_iter) or \
            (manager.iteration % self.interval == 0)
