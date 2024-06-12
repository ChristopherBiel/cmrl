# general
import numpy as np

# for logging
import pickle

# utilities
from helper import *

# data directory
data_dir = './results/'


# specify learning case if not using baseline
n_hidden = 50
linear_type = "linear"  # linear, linear_bjorck, linear_spectral_norm
activation_type = "tanh"  # tanh, relu, groupsort
l_constant = 0.1  # 1 # 5e-3
learning_rate = 1e-2
add_disturbance_flag = True

# file directories
baseline_file_dir = data_dir + 'baseline'
learning_file_dir = data_dir + 'learning_' + str(n_hidden) + linear_type + '_' + activation_type + '_l' + str(l_constant) + '_lr1e' + str(int(np.log10(learning_rate)))
if add_disturbance_flag:
    # baseline_file_dir += '_disturbed'
    learning_file_dir += '_disturbed'
baseline_file_dir += '.pkl'
learning_file_dir += '.pkl'
print('baseline dir: ', learning_file_dir)
print('learning dir: ', learning_file_dir)

# import data
with open(baseline_file_dir, 'rb') as f:
    log_times_baseline, log_state_baseline, log_reference_baseline, log_desired_baseline, log_error_baseline, rmse_x_baseline = pickle.load(f)

with open(learning_file_dir, 'rb') as f:
    log_times_learning, log_state_learning, log_reference_learning, log_desired_learning, log_error_learning, rmse_x_learning = pickle.load(f)

print('rmse baseline: %.2f' % rmse_x_baseline)
print('rmse learning: %.2f' % rmse_x_learning)

# plot comparison
save_dir = data_dir + 'learning_' + str(n_hidden) + linear_type + '_' + activation_type + '_l' + str(l_constant) + '_lr1e' + str(int(np.log10(learning_rate)))
if add_disturbance_flag:
    save_dir += '_disturbed'
save_dir += '.png'
plot_tracking_comparison(log_times_baseline, log_state_baseline, log_desired_baseline, log_times_learning, log_state_learning, log_reference_learning, log_desired_learning, save_dir)