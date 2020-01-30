import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        #initialize all members you need

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience

