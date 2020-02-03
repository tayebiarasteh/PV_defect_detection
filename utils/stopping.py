import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0


    def step(self, current_loss, best_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss >= best_loss:
            self.counter += 1
        else:
            best_loss = current_loss
            self.counter = 0

        return best_loss


    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.counter >= self.patience
