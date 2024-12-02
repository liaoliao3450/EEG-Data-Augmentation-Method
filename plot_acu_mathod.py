import os
import numpy as np
from scipy.stats import multivariate_normal
import mne
from matplotlib import pyplot as plt



# 'Deep4Net',, 'ShallowNet''aug_noise', 'aug_smooth_time_mask','aug_channels_symmetry', 'aug_sign_flip', 'aug_ft_surrogate','aug_channels_shuffle', 'aug_frequency_shift'
######## 单倍增强数据 9*5
# acc_smooth_time_mask_eegnet = np.load(f'./deep_learning/acc_smooth_time_mask_eegnet.npy')
# acc_sign_flip_eegnet = np.load(f'./deep_learning/acc_sign_flip_eegnet.npy')
# acc_channels_symmetry_eegnet = np.load(f'./deep_learning/acc_channels_symmetry_eegnet.npy')
# acc_channels_shuffle_eegnet = np.load(f'./deep_learning/acc_channels_shuffle_eegnet.npy')
# acc_noise_eegnet = np.load(f'./deep_learning/acc_noise_eegnet.npy')
# acc_ft_surrogate_eegnet = np.load(f'./deep_learning/acc_ft_surrogate_eegnet.npy')
# acc_frequency_shift_eegnet = np.load(f'./deep_learning/acc_frequency_shift_eegnet.npy')

acc_EEGNet_original = np.load(f'./deep_learning/acc_EEGNet_original.npy')
average_acc_EEGNet_original = np.mean(acc_EEGNet_original)
acc_Deep4Net_original = np.load(f'./deep_learning/acc_Deep4Net_original.npy')
average_acc_Deep4Net_original = np.mean(acc_Deep4Net_original)
acc_ShallowNet_original = np.load(f'./deep_learning/acc_ShallowNet_original.npy')
average_acc_ShallowNet_original = np.mean(acc_ShallowNet_original)
######## 单倍增强数据 9 ranon state =42
# ACC_EEGNet_aug_noise = np.load(f'./deep_learning/ACC_EEGNet_aug_noise.npy')
# ACC_EEGNet_aug_smooth_time_mask = np.load(f'./deep_learning/ACC_EEGNet_aug_smooth_time_mask.npy')
# ACC_EEGNet_aug_channels_symmetry = np.load(f'./deep_learning/ACC_EEGNet_aug_channels_symmetry.npy')
# ACC_EEGNet_aug_sign_flip = np.load(f'./deep_learning/ACC_EEGNet_aug_sign_flip.npy')
# ACC_EEGNet_aug_ft_surrogate = np.load(f'./deep_learning/ACC_EEGNet_aug_ft_surrogate.npy')
# ACC_EEGNet_aug_channels_shuffle = np.load(f'./deep_learning/ACC_EEGNet_aug_channels_shuffle.npy')
# ACC_EEGNet_aug_frequency_shift = np.load(f'./deep_learning/ACC_EEGNet_aug_frequency_shift.npy')
# ACC_EEGNet_aug_gmm = np.load(f'./deep_learning/ACC_EEGNet_aug_gmm.npy')
#
# ACC_Deep4Net_aug_noise = np.load(f'./deep_learning/ACC_Deep4Net_aug_noise.npy')
# ACC_Deep4Net_aug_smooth_time_mask = np.load(f'./deep_learning/ACC_Deep4Net_aug_smooth_time_mask.npy')
# ACC_Deep4Net_aug_channels_symmetry = np.load(f'./deep_learning/ACC_Deep4Net_aug_channels_symmetry.npy')
# ACC_Deep4Net_aug_sign_flip = np.load(f'./deep_learning/ACC_Deep4Net_aug_sign_flip.npy')
# ACC_Deep4Net_aug_ft_surrogate = np.load(f'./deep_learning/ACC_Deep4Net_aug_ft_surrogate.npy')
# ACC_Deep4Net_aug_channels_shuffle = np.load(f'./deep_learning/ACC_Deep4Net_aug_channels_shuffle.npy')
# ACC_Deep4Net_aug_frequency_shift = np.load(f'./deep_learning/ACC_Deep4Net_aug_frequency_shift.npy')
# ACC_Deep4Net_aug_gmm = np.load(f'./deep_learning/ACC_Deep4Net_aug_gmm.npy')
#
# ACC_ShallowNet_aug_noise = np.load(f'./deep_learning/ACC_ShallowNet_aug_noise.npy')
# ACC_ShallowNet_aug_smooth_time_mask = np.load(f'./deep_learning/ACC_ShallowNet_aug_smooth_time_mask.npy')
# ACC_ShallowNet_aug_channels_symmetry = np.load(f'./deep_learning/ACC_ShallowNet_aug_channels_symmetry.npy')
# ACC_ShallowNet_aug_sign_flip = np.load(f'./deep_learning/ACC_ShallowNet_aug_sign_flip.npy')
# ACC_ShallowNet_aug_ft_surrogate = np.load(f'./deep_learning/ACC_ShallowNet_aug_ft_surrogate.npy')
# ACC_ShallowNet_aug_channels_shuffle = np.load(f'./deep_learning/ACC_ShallowNet_aug_channels_shuffle.npy')
# ACC_ShallowNet_aug_frequency_shift = np.load(f'./deep_learning/ACC_ShallowNet_aug_frequency_shift.npy')
# ACC_ShallowNet_aug_gmm = np.load(f'./deep_learning/ACC_ShallowNet_aug_gmm.npy')

####### 单倍增强数据和原始数据 9
ACCC_EEGNet_aug_noise = np.load(f'./deep_learning/ACCC_EEGNet_aug_noise.npy')
average_ACCC_EEGNet_aug_noise = np.mean(ACCC_EEGNet_aug_noise)
ACCC_EEGNet_aug_smooth_time_mask = np.load(f'./deep_learning/ACCC_EEGNet_aug_smooth_time_mask.npy')
average_ACCC_EEGNet_aug_smooth_time_mask = np.mean(ACCC_EEGNet_aug_smooth_time_mask)
ACCC_EEGNet_aug_channels_symmetry = np.load(f'./deep_learning/ACCC_EEGNet_aug_channels_symmetry.npy')
average_ACCC_EEGNet_aug_channels_symmetry  = np.mean(ACCC_EEGNet_aug_channels_symmetry )
ACCC_EEGNet_aug_sign_flip = np.load(f'./deep_learning/ACCC_EEGNet_aug_sign_flip.npy')
average_ACC_EEGNet_aug_sign_flip = np.mean(ACCC_EEGNet_aug_sign_flip)
ACCC_EEGNet_aug_ft_surrogate = np.load(f'./deep_learning/ACCC_EEGNet_aug_ft_surrogate.npy')
average_EEGNet_aug_ft_surrogate = np.mean(ACCC_EEGNet_aug_ft_surrogate)
ACCC_EEGNet_aug_channels_shuffle = np.load(f'./deep_learning/ACCC_EEGNet_aug_channels_shuffle.npy')
average_ACCC_EEGNet_aug_channels_shuffle = np.mean(ACCC_EEGNet_aug_channels_shuffle)
ACCC_EEGNet_aug_frequency_shift = np.load(f'./deep_learning/ACCC_EEGNet_aug_frequency_shift.npy')
average_ACCC_EEGNet_aug_frequency_shift = np.mean(ACCC_EEGNet_aug_frequency_shift)
ACCC_EEGNet_aug_gmm = np.load(f'./deep_learning/ACCC_EEGNet_aug_gmm.npy')
average_ACCC_EEGNet_aug_gmm = np.mean(ACCC_EEGNet_aug_gmm)

ACCC_Deep4Net_aug_noise = np.load(f'./deep_learning/ACC_Deep4Net_aug_noise.npy')
average_ACCC_Deep4Net_aug_noise = np.mean(ACCC_Deep4Net_aug_noise)
ACCC_Deep4Net_aug_smooth_time_mask = np.load(f'./deep_learning/ACCC_Deep4Net_aug_smooth_time_mask.npy')
average_ACCC_Deep4Net_aug_smooth_time_mask = np.mean(ACCC_Deep4Net_aug_smooth_time_mask)
ACCC_Deep4Net_aug_channels_symmetry = np.load(f'./deep_learning/ACCC_Deep4Net_aug_channels_symmetry.npy')
average_ACCC_Deep4Net_aug_channels_symmetry = np.mean(ACCC_Deep4Net_aug_channels_symmetry)
ACCC_Deep4Net_aug_sign_flip = np.load(f'./deep_learning/ACCC_Deep4Net_aug_sign_flip.npy')
average_ACCC_Deep4Net_aug_sign_flip = np.mean(ACCC_Deep4Net_aug_sign_flip)
ACCC_Deep4Net_aug_ft_surrogate = np.load(f'./deep_learning/ACCC_Deep4Net_aug_ft_surrogate.npy')
average_ACCC_Deep4Net_aug_ft_surrogate = np.mean(ACCC_Deep4Net_aug_ft_surrogate)
ACCC_Deep4Net_aug_channels_shuffle = np.load(f'./deep_learning/ACCC_Deep4Net_aug_channels_shuffle.npy')
average_ACCC_Deep4Net_aug_channels_shuffle = np.mean(ACCC_Deep4Net_aug_channels_shuffle)
ACCC_Deep4Net_aug_frequency_shift = np.load(f'./deep_learning/ACCC_Deep4Net_aug_frequency_shift.npy')
average_ACCC_Deep4Net_aug_frequency_shift = np.mean(ACCC_Deep4Net_aug_frequency_shift)
ACCC_Deep4Net_aug_gmm = np.load(f'./deep_learning/ACCC_Deep4Net_aug_gmm.npy')
average_ACCC_Deep4Net_aug_gmm= np.mean(ACCC_Deep4Net_aug_gmm)

ACCC_ShallowNet_aug_noise = np.load(f'./deep_learning/ACCC_ShallowNet_aug_noise.npy')
average_ACCC_ShallowNet_aug_noise= np.mean(ACCC_ShallowNet_aug_noise)
ACCC_ShallowNet_aug_smooth_time_mask = np.load(f'./deep_learning/ACCC_ShallowNet_aug_smooth_time_mask.npy')
average_ACCC_ShallowNet_aug_smooth_time_mask = np.mean(ACCC_ShallowNet_aug_smooth_time_mask)
ACCC_ShallowNet_aug_channels_symmetry = np.load(f'./deep_learning/ACCC_ShallowNet_aug_channels_symmetry.npy')
average_ACCC_ShallowNet_aug_channels_symmetry = np.mean(ACCC_ShallowNet_aug_channels_symmetry)
ACCC_ShallowNet_aug_sign_flip = np.load(f'./deep_learning/ACCC_ShallowNet_aug_sign_flip.npy')
average_ACCC_ShallowNet_aug_sign_flip = np.mean(ACCC_ShallowNet_aug_sign_flip)
ACCC_ShallowNet_aug_ft_surrogate = np.load(f'./deep_learning/ACCC_ShallowNet_aug_ft_surrogate.npy')
average_ACCC_ShallowNet_aug_ft_surrogate = np.mean(ACCC_ShallowNet_aug_ft_surrogate)
ACCC_ShallowNet_aug_channels_shuffle = np.load(f'./deep_learning/ACCC_ShallowNet_aug_channels_shuffle.npy')
average_ACCC_ShallowNet_aug_channels_shuffle = np.mean(ACCC_ShallowNet_aug_channels_shuffle)
ACCC_ShallowNet_aug_frequency_shift = np.load(f'./deep_learning/ACCC_ShallowNet_aug_frequency_shift.npy')
average_ACCC_ShallowNet_aug_frequency_shift = np.mean(ACCC_ShallowNet_aug_frequency_shift)
ACCC_ShallowNet_aug_gmm = np.load(f'./deep_learning/ACCC_ShallowNet_aug_gmm.npy')
average_ACCC_ShallowNet_aug_gmm = np.mean(ACCC_ShallowNet_aug_gmm)

ACCC_FBCSP_aug_noise = np.load(f'./deep_learning/ACCC_FBCSP_aug_noise.npy')
average_FBCSP_aug_noise = np.mean(ACCC_FBCSP_aug_noise)
ACCC_FBCSP_aug_smooth_time_mask = np.load(f'./deep_learning/ACCC_FBCSP_aug_smooth_time_mask.npy')
average_ACCC_FBCSP_aug_smooth_time_mask = np.mean(ACCC_FBCSP_aug_smooth_time_mask)
ACCC_FBCSP_aug_channels_symmetry = np.load(f'./deep_learning/ACCC_FBCSP_aug_channels_symmetry.npy')
average_ACCC_FBCSP_aug_channels_symmetry = np.mean(ACCC_FBCSP_aug_channels_symmetry)
ACCC_FBCSP_aug_sign_flip = np.load(f'./deep_learning/ACCC_FBCSP_aug_sign_flip.npy')
average_ACCC_FBCSP_aug_sign_flip = np.mean(ACCC_FBCSP_aug_sign_flip)
ACCC_FBCSP_aug_ft_surrogate = np.load(f'./deep_learning/ACCC_FBCSP_aug_ft_surrogate.npy')
average_ACCC_FBCSP_aug_ft_surrogate = np.mean(ACCC_FBCSP_aug_ft_surrogate)
ACCC_FBCSP_aug_channels_shuffle = np.load(f'./deep_learning/ACCC_FBCSP_aug_channels_shuffle.npy')
average_ACCC_FBCSP_aug_channels_shuffle = np.mean(ACCC_FBCSP_aug_channels_shuffle)
ACCC_FBCSP_aug_frequency_shift = np.load(f'./deep_learning/ACCC_FBCSP_aug_frequency_shift.npy')
average_ACCC_FBCSP_aug_frequency_shift = np.mean(ACCC_FBCSP_aug_frequency_shift)
ACCC_FBCSP_aug_bandstop_filter = np.load(f'./deep_learning/ACCC_FBCSP_aug_bandstop_filter.npy')
average_ACCC_FBCSP_aug_bandstop_filter= np.mean(ACCC_FBCSP_aug_bandstop_filter)
ACCC_FBCSP_aug_time_reverse = np.load(f'./deep_learning/ACCC_FBCSP_aug_time_reverse.npy')
average_ACCC_FBCSP_aug_time_reverse = np.mean(ACCC_FBCSP_aug_time_reverse)
ACCC_FBCSP_aug_gmm_transform = np.load(f'./deep_learning/ACCC_FBCSP_aug_gmm_transform.npy')
average_ACCC_FBCSP_aug_gmm_transform = np.mean(ACCC_FBCSP_aug_gmm_transform)


acc_LSTM_aug_gmm_transform = np.load(f'./deep_learning/acc_LSTM_aug_gmm_transform.npy')
average_acc_LSTM_aug_gmm_transform = np.mean(acc_LSTM_aug_gmm_transform)
acc_LSTM_aug_time_reverse = np.load(f'./deep_learning/acc_LSTM_aug_time_reverse.npy')
average_acc_LSTM_aug_time_reverse = np.mean(acc_LSTM_aug_time_reverse)
acc_LSTM_aug_bandstop_filter = np.load(f'./deep_learning/acc_LSTM_aug_bandstop_filter.npy')
average_acc_LSTM_aug_bandstop_filter = np.mean(acc_LSTM_aug_bandstop_filter)

acc_Deep4Net_aug_gmm_transform = np.load(f'./deep_learning/acc_Deep4Net_aug_gmm_transform.npy')
average_acc_Deep4Net_aug_gmm_transform = np.mean(acc_Deep4Net_aug_gmm_transform)
acc_Deep4Net_aug_time_reverse = np.load(f'./deep_learning/acc_Deep4Net_aug_time_reverse.npy')
average_acc_Deep4Net_aug_time_reverse = np.mean(acc_Deep4Net_aug_time_reverse)
acc_Deep4Net_aug_bandstop_filter = np.load(f'./deep_learning/acc_Deep4Net_aug_bandstop_filter.npy')
average_acc_Deep4Net_aug_bandstop_filter = np.mean(acc_Deep4Net_aug_bandstop_filter)

acc_EEGNet_aug_gmm_transform = np.load(f'./deep_learning/acc_EEGNet_aug_gmm_transform.npy')
average_acc_EEGNet_aug_gmm_transform = np.mean(acc_EEGNet_aug_gmm_transform)
acc_EEGNet_aug_time_reverse = np.load(f'./deep_learning/acc_EEGNet_aug_time_reverse.npy')
average_acc_EEGNet_aug_time_reverse = np.mean(acc_EEGNet_aug_time_reverse)
acc_EEGNet_aug_bandstop_filter = np.load(f'./deep_learning/acc_EEGNet_aug_bandstop_filter.npy')
average_acc_EEGNet_aug_bandstop_filter = np.mean(acc_EEGNet_aug_bandstop_filter)

acc_ShallowNet_aug_gmm_transform = np.load(f'./deep_learning/acc_ShallowNet_aug_gmm_transform.npy')
average_acc_ShallowNet_aug_gmm_transform = np.mean(acc_ShallowNet_aug_gmm_transform)
acc_ShallowNet_aug_time_reverse = np.load(f'./deep_learning/acc_ShallowNet_aug_time_reverse.npy')
average_acc_ShallowNet_aug_time_reverse = np.mean(acc_ShallowNet_aug_time_reverse)
acc_ShallowNet_aug_bandstop_filter = np.load(f'./deep_learning/acc_ShallowNet_aug_bandstop_filter.npy')
average_acc_ShallowNet_aug_bandstop_filter = np.mean(acc_ShallowNet_aug_bandstop_filter)


acc_list_original = acc_Deep4Net_original
# acc_list_original = acc_list_original.mean(axis=1)
acc_list_generate = ACC_EEGNet_aug_noise
# acc_list_generate = acc_list_generate.mean(axis=1)

# acc_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

labels = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9']

x = np.arange(len(labels))
width = 0.30

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, acc_list_original, width, label='Original')
rects2 = ax.bar(x + width/2, acc_list_generate, width, label='Generate')

ax.set_ylabel('Subject')
ax.set_title('Bar chart comparing the classification accuracy of two groups of nine subjects')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()