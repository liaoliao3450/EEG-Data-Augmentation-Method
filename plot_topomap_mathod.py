import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import Normalizer
from matplotlib.colors import Normalize

# 设置临时文件路径为C:\Temp
temp_dir = "C:\\Temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['MNE_DATA'] = temp_dir
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir


# 创建info对象
def create_raw(data, ch_names, sfreq, ch_types):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

# 设置标准10-20电极系统布局
montage = mne.channels.make_standard_montage("standard_1020")

# 计算PSD并归一化
def compute_psd(raw, sfreq, fmin=1, fmax=40):
    psds, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    return psds, freqs

# 加载数据
data = np.load('data/original_data_1.npy')
aug_noise_data = np.load('data/aug_noise_data_1.npy')
aug_smooth_time_mask_data = np.load('data/aug_smooth_time_mask_data_1.npy')
aug_channels_symmetry_data = np.load('data/aug_channels_symmetry_data_1.npy')
aug_sign_flip_data = np.load('data/aug_sign_flip_data_1.npy')
aug_ft_surrogate_data = np.load('data/aug_ft_surrogate_data_1.npy')
aug_channels_shuffle_data = np.load('data/aug_channels_shuffle_data_1.npy')
aug_frequency_shift_data = np.load('data/aug_frequency_shift_data_1.npy')
aug_gmm_data = np.load('data/aug_gmm_data_1.npy')
original_labels = np.load('data/original_labels_11.npy')

# 数据列表
data_list = [
    (data, "Original Data"),
    (aug_noise_data, "Noise"),
    (aug_smooth_time_mask_data, "Smooth Time Mask"),
    (aug_channels_symmetry_data, "Channels Symmetry"),
    (aug_sign_flip_data, "Sign Flip"),
    (aug_ft_surrogate_data, "FT Surrogate"),
    (aug_channels_shuffle_data, "Channels Shuffle"),
    (aug_frequency_shift_data, "Frequency Shift"),
    (aug_gmm_data, "GMM")
]
# 生成假数据作为示例
# data = np.random.randn(22, 2500)
for i in range(len(data_list)):
    data_list[i] = (data_list[i][0][0].reshape(22, -1), data_list[i][1])

ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
            'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_types = ['eeg'] * 22
sfreq = 250  # Hz


avg_psds = []

freq_ranges = [(0, 4), (4, 8), (8, 13), (13, 30)]

# 创建九排四张图的布局
fig, axs = plt.subplots(9, len(freq_ranges), figsize=(15, 10))


# 创建归一化对象
norm = Normalizer()


for j, (current_data, label) in enumerate(data_list):

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    current_raw = mne.io.RawArray(current_data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    current_raw.set_montage(montage)
    current_psds, freqs = compute_psd(current_raw, sfreq)

    for i, (low, high) in enumerate(freq_ranges):
        # 选择频率范围并计算平均PSD
        freq_idx = np.where((freqs >= low) & (freqs <= high))[0]
        avg_psds = np.mean(current_psds[:, freq_idx], axis=1)
        # norm_avg_psds = norm.fit_transform(avg_psds.reshape(-1, 1)).flatten()  # 应用归一化对象
        # im, _ = mne.viz.plot_topomap(norm_avg_psds, current_raw.info, axes=axs[j , i], show=False, cmap='bwr')
        im, _ = mne.viz.plot_topomap(avg_psds, current_raw.info, axes=axs[j , i], show=False, cmap='bwr')
        cbar = plt.colorbar(im, ax=axs[j , i], fraction=0.046, pad=0.04)
        # cbar.set_label('Normalized PSD', fontsize=12)
        # axs[j , i].set_title(f' {low}-{high} Hz', fontsize=12)
        axs[j , i].set_xlabel(f'{label}_topomap', fontsize=12)

plt.tight_layout()
plt.show()

# 绘制地形图
# fig, ax = plt.subplots()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# # 绘制电极位置图
# mne.viz.plot_topomap(avg_psds, raw.info, axes=ax, show=False)

# 绘制电极位置图并添加色彩条
# im, _ = mne.viz.plot_topomap(avg_psds, raw.info, axes=ax, show=False)
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label('Normalized PSD')

# 绘制电极位置图并添加色彩条
# im1, _ = mne.viz.plot_topomap(avg_psds, raw.info, axes=ax1, show=False)
# cbar1 = plt.colorbar(im1, ax=ax1)
# cbar1.set_label('Normalized PSD')
# ax1.set_xlabel('Original_Data_topomap')

# 绘制电极位置图并添加色彩条
# im2, _ = mne.viz.plot_topomap(avg_psds_finaldata, raw_finaldata.info, axes=ax2, show=False)
# cbar2 = plt.colorbar(im2, ax=ax2)
# cbar2.set_label('Normalized PSD')
# ax2.set_xlabel('Generate_Data_topomap')
#
# plt.show()
