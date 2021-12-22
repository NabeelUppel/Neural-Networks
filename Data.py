import numpy as np
import matplotlib.pyplot as plt
import math


class DataSetup:
    def __init__(self, size):
        self.Data = self.setup(size)

    def setup(self, size):
        f1 = []
        f2 = []
        labels = []
        for x in range(size):
            x1 = np.random.uniform(0, 1)
            f1.append(x1)
            x2 = np.random.uniform(0, 1)
            f2.append(x2)
            l = self.labelGen(x1, x2)
            labels.append(l)

        f1 = np.array(f1)
        f2 = np.array(f2)
        labels = np.array(labels)
        l1 = np.array([1] * size)

        d = np.column_stack([f1, f2, labels])
        return d

    def function(self, x):
        f = (x ** 2) * math.sin(2 * math.pi * x) + 0.7
        return f

    def labelGen(self, x1, x2):
        f = self.function(x1)
        if f > x2:
            return 0
        else:
            return 1

    def graph(self):
        C0 = self.Data[self.Data[:, 2] == 0]
        C1 = self.Data[self.Data[:, 2] == 1]
        x0 = C0[:, 0]
        y0 = C0[:, 1]
        x1 = C1[:, 0]
        y1 = C1[:, 1]

        f, ax = plt.subplots()
        ax.scatter(x0, y0, color="teal")
        ax.scatter(x1, y1, color="red")
        plt.show()
        return ax
