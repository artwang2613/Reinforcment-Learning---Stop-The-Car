import numpy as np

def increment_counter(counter, maxDigit):
    for i in list(range(len(counter)))[::-1]:
        counter[i] += 1
        if (counter[i] > maxDigit):
            counter[i] = 0
        else:
            break

class FourierBasis(object):
    def __init__(self, ranges, order, both=False):
        self.iorder = int(order)
        self.dorder = int(order)
        self.ranges = ranges.astype(np.float64)
        self.feat_range = (ranges[:, 1] - ranges[:, 0]).astype(np.float64)
        self.feat_range[self.feat_range == 0] = 1


        iTerms = order * ranges.shape[0]  # number independent terms
        dTerms = pow(order+1, ranges.shape[0])  # number of dependent
        oTerms = min(order, order) * ranges.shape[0]  # number of overlap terms
        self.num_features = int(iTerms + dTerms - oTerms)

        self.both = both
        self.num_input_features = int(ranges.shape[0])
        self.C = np.zeros((self.num_features, ranges.shape[0]), dtype=np.float64)
        counter = np.zeros(ranges.shape[0])
        termCount = 0
        while termCount < dTerms:
            for i in range(ranges.shape[0]):
                self.C[termCount, i] = counter[i]
            increment_counter(counter, order)
            termCount += 1
        for i in range(ranges.shape[0]):
            for j in range(order+1, order+1):
                self.C[termCount, i] = j
                termCount += 1

        self.C = self.C.T * np.pi
        if both:
            self.num_features *= 2

    def encode(self, x):
        x = x.flatten().astype(np.float64)
        scaled = (x - self.ranges[:, 0]) / self.feat_range
        dprod = np.dot(scaled, self.C)
        if self.both:
            basis = np.concatenate([np.cos(dprod), np.sin(dprod)])
        else:
            basis = np.cos(dprod)
        return basis

    def getNumFeatures(self):
        return self.num_features