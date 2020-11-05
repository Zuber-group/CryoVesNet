import numpy as np


class confusionMatrix:
    def __init__(self, prediction, reference):
        self.tp = np.sum(np.logical_and(prediction == 1, reference == 1))
        self.tn = np.sum(np.logical_and(prediction == 0, reference == 0))
        self.fp = np.sum(np.logical_and(prediction == 1, reference == 0))
        self.fn = np.sum(np.logical_and(prediction == 0, reference == 1))
        self.n = prediction.size

    def accuracy(self):
        sum_cm = self.tp + self.tn + self.fp + self.fn

        if sum_cm != 0:
            return (self.tp + self.tn) / sum_cm
        else:
            return 0

    def dice_metric(self):
        if (self.tp == 0) and \
                ((self.tp + self.fp + self.fn) == 0):
            return 1.
        return 2 * self.tp / \
               (2 * self.tp + self.fp + self.fn)

    def jaccard_index(self):
        if (self.tp == 0) and \
                ((self.tp + self.fp + self.fn) == 0):
            return 1.
        return self.tp / \
               (self.tp + self.fp + self.fn)

    def __str__(self):
        return("true positive: "+str(self.tp)+"\ntrue negative: " + str(self.tn) +"\nfalse positive: " + str(self.fp) + "\nfalse negative: " + str(self.fn))




