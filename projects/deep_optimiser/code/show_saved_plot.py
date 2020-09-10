

import pickle
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from matplotlib import pyplot as plt


#filename = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-13_14:14:16/test_vis/model.05-0.0936.hdf5_test_gaussian_zero_init/2020-05-16 16:44:39.137867.pickle"
#filename = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_11:11:43/test_vis/model.999-0.0169.hdf5_test_gaussian_zero_init/2020-05-21 19:21:39.044301.pickle"
#filename = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_11:12:22/test_vis/model.1099-0.0270.hdf5_test_gaussian_zero_init/2020-05-21 19:18:25.797403.pickle"
#filename= "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_11:13:14/test_vis/model.999-0.0385.hdf5_test_gaussian_zero_init/2020-05-21 19:17:59.721974.pickle"
#filename="/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_11:15:02/test_vis/model.999-0.0096.hdf5_test_gaussian_zero_init/2020-05-21 19:22:42.241396.pickle"
#filename="/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_11:08:37/test_vis/model.1099-0.0189.hdf5_test_gaussian_zero_init/2020-05-21 19:24:38.831773.pickle"
#filename="/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-19_14:18:02/test_vis/model.549-0.0623.hdf5_test_gaussian_zero_init/2020-05-21 19:27:08.361375.pickle"
#filename="/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-14_13:20:10/test_vis/model.1149-0.0465.hdf5_test_gaussian_zero_init/2020-05-17 08:47:14.307438.pickle"
filename="/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-20_13:51:14/test_vis/model.1099-0.0409.hdf5_test_gaussian_zero_init/2020-05-21 19:43:00.523999.pickle"
fig = pickle.load(open(filename, "r"))

#fig.show()
plt.show()

