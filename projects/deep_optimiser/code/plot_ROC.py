
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



sgd_ROC = np.load("sgd_ROC.npy")
adam_ROC = np.load("adam_ROC.npy")
deep_opt_ROC = np.load("deep_opt_ROC.npy")


bins = np.linspace(0, 1, num=100)
plt.plot(bins, sgd_ROC, label="SGD", color='r')
plt.plot(bins, adam_ROC, label="Adam", color='g')
plt.plot(bins, deep_opt_ROC, label="Deep opt.", color='b')

plt.xlabel("Angular error (radians)", fontsize=16, fontweight="bold")
plt.ylabel("Percentage of samples", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend(prop={'size': 14, 'weight': 'bold'})
plt.show()


