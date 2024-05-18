# Equivalent of `TensorFlow`'s EarlyStopping and ReduceLROnPlateau callbacks
class EarlyStopping:
    def __init__(self, patience=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False

    def step(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class ReduceLROnPlateau:
    def __init__(self, factor=0.1, patience=10):
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def step(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.factor *= 0.1
                self.counter = 0
        else:
            self.best_score = val_loss