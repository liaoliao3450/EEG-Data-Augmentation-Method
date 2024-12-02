
import numpy as np
from matplotlib import pyplot as plt
from plot_utils import FONTSIZE
from plot_utils import plot_signal

from braindecode.augmentation.transforms import GaussianNoise, time_reverse, SignFlip, FTSurrogate
from braindecode.augmentation.transforms import ChannelsShuffle, FrequencyShift, BandstopFilter, SmoothTimeMask, ChannelsSymmetry



# 定义数据增强的概率
probability = 1
random_state =22
# 每个增强实例会以给定概率应用于输入数据
noise_transform = GaussianNoise(probability=probability, std=0.3,random_state=22)  # 添加高斯噪声
# time_reverse_transform = time_reverse()      # 时间反转
sign_flip_transform = SignFlip(probability=probability,random_state=22)            # 符号翻转
ft_surrogate_transform = FTSurrogate(probability=probability,random_state=22)      # 傅里叶变换代理
channels_shuffle_transform = ChannelsShuffle(probability=probability,random_state=13)  # 通道随机打乱
frequency_shift_transform = FrequencyShift(probability=probability, sfreq=5.0,random_state=22)  # 频率偏移
# bandstop_filter_transform = BandstopFilter(probability=probability, band_freq=(49, 51))  # 带阻滤波
smooth_time_mask_transform = SmoothTimeMask(probability=probability, mask_len_samples=100,random_state=22)  # 时间掩码
channels_symmetry_transform = ChannelsSymmetry(probability=probability, ordered_ch_names=[f'Ch{i+1}' for i in range(22)],random_state=22)  # 通道对称增强

data = np.load('data/original_data_1.npy')
# aug_noise_data = np.load('data/aug_noise_data_1.npy')
# aug_smooth_time_mask_data = np.load('data/aug_smooth_time_mask_data_1.npy')
# aug_channels_symmetry_data = np.load('data/aug_channels_symmetry_data_1.npy')
# aug_sign_flip_data = np.load('data/aug_sign_flip_data_1.npy')
# aug_ft_surrogate_data = np.load('data/aug_ft_surrogate_data_1.npy')
# aug_channels_shuffle_data = np.load('data/aug_channels_shuffle_data_1.npy')
# aug_frequency_shift_data = np.load('data/aug_frequency_shift_data_1.npy')
aug_bandstop_filter_data = np.load('data/aug_bandstop_filter_data_1.npy')
aug_time_reverse_data  = np.load('data/aug_time_reverse_data_1.npy')
aug_gmm_data = np.load('data/aug_gmm_data_1.npy')
original_labels = np.load('data/original_labels_11.npy')

index_left_hand_data = 3
data_trans = data[index_left_hand_data:index_left_hand_data+2]
# data_trans = data_trans.reshape((2, 22, 751))
aug_noise = noise_transform(data_trans)
    # aug_time_reverse= time_reverse_transform(data)
aug_sign_flip = sign_flip_transform(data_trans)
aug_ft_surrogate = ft_surrogate_transform(data_trans)
aug_channels_shuffle = channels_shuffle_transform(data_trans)
aug_frequency_shift = frequency_shift_transform(data_trans)
    # aug_bandstop_filter = bandstop_filter_transform(data)
aug_smooth_time_mask = smooth_time_mask_transform(data_trans)
aug_channels_symmetry = channels_symmetry_transform(data_trans)

windows = data   # 原始数据

# labels = get_labels(windows)
N3_indices_left_hand = np.where(original_labels == 1)  # left_hand
N3_indices_right_hand = np.where(original_labels == 2)  # right_hand
N3_indices_foot = np.where(original_labels == 3)  # foot
N3_indices_tongue = np.where(original_labels == 4)  # tongue
channel = 21
# %%  索引
trial =index_left_hand_data+1
# index_left_hand = N3_indices_left_hand[0][trial]  # 第1行第15个trial
index_left_hand = 0
# index_right_hand = N3_indices_right_hand[0][trial]
# index_foot = N3_indices_foot[0][trial ]
# index_tongue = N3_indices_tongue[0][trial]
#************************************* 可视化数据
window_test_left_hand = windows[index_left_hand_data,channel,:]    # 第1行第15个trial的22通道数据
# window_test_right_hand = windows[index_right_hand,channel,:]
# window_test_foot = windows[index_foot,channel,:]
# window_test_tongue = windows[index_tongue,channel,:]

#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
# windows_aug_noise_data = aug_noise_data  # 生成数据1
windows_aug_noise_data = aug_noise
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_noise_data_left_hand = windows_aug_noise_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_noise_data_right_hand = windows_aug_noise_data[index_right_hand,channel,:]
# windows_aug_noise_data_foot = windows_aug_noise_data[index_foot,channel,:]
# windows_aug_noise_data_tongue = windows_aug_noise_data[index_tongue,channel,:]
# %%
# windows_aug_smooth_time_mask_data = aug_smooth_time_mask_data  # 生成数据2
windows_aug_smooth_time_mask_data = aug_smooth_time_mask # 生成数据2
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_smooth_time_mask_data_left_hand = windows_aug_smooth_time_mask_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_smooth_time_mask_data_right_hand = windows_aug_smooth_time_mask_data[index_right_hand,channel,:]
# windows_aug_smooth_time_mask_data_foot = windows_aug_smooth_time_mask_data[index_foot,channel,:]
# windows_aug_smooth_time_mask_data_tongue = windows_aug_smooth_time_mask_data[index_tongue,channel,:]
# %%
# windows_aug_channels_symmetry_data = aug_channels_symmetry_data  # 生成数据3
windows_aug_channels_symmetry_data = aug_channels_symmetry # 生成数据3
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_channels_symmetry_data_left_hand = windows_aug_channels_symmetry_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_channels_symmetry_data_right_hand = windows_aug_channels_symmetry_data[index_right_hand,channel,:]
# windows_aug_channels_symmetry_data_foot = windows_aug_channels_symmetry_data[index_foot,channel,:]
# windows_aug_channels_symmetry_data_tongue = windows_aug_channels_symmetry_data[index_tongue,channel,:]

# windows_aug_sign_flip_data = aug_sign_flip_data  # 生成数据4
windows_aug_sign_flip_data = aug_sign_flip  # 生成数据4
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_sign_flip_data_left_hand = windows_aug_sign_flip_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_sign_flip_data_right_hand = windows_aug_sign_flip_data[index_right_hand,channel,:]
# windows_aug_sign_flip_data_foot = windows_aug_sign_flip_data[index_foot,channel,:]
# windows_aug_sign_flip_data_tongue = windows_aug_sign_flip_data[index_tongue,channel,:]

# windows_aug_ft_surrogate_data = aug_ft_surrogate_data  # 生成数据5
windows_aug_ft_surrogate_data = aug_ft_surrogate # 生成数据5
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_ft_surrogate_data_left_hand = windows_aug_ft_surrogate_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_ft_surrogate_data_right_hand = windows_aug_ft_surrogate_data[index_right_hand,channel,:]
# windows_aug_ft_surrogate_data_foot = windows_aug_ft_surrogate_data[index_foot,channel,:]
# windows_aug_ft_surrogate_data_tongue = windows_aug_ft_surrogate_data[index_tongue,channel,:]

# windows_aug_channels_shuffle_data = aug_channels_shuffle_data  # 生成数据6
windows_aug_channels_shuffle_data = aug_channels_shuffle  # 生成数据6
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_channels_shuffle_data_left_hand = windows_aug_channels_shuffle_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_channels_shuffle_data_right_hand = windows_aug_channels_shuffle_data[index_right_hand,channel,:]
# windows_aug_channels_shuffle_data_foot = windows_aug_channels_shuffle_data[index_foot,channel,:]
# windows_aug_channels_shuffle_data_tongue = windows_aug_channels_shuffle_data[index_tongue,channel,:]

# windows_aug_frequency_shift_data = aug_frequency_shift_data  # 生成数据7
windows_aug_frequency_shift_data = aug_frequency_shift  # 生成数据7
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_frequency_shift_data_left_hand = windows_aug_frequency_shift_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_frequency_shift_data_right_hand = windows_aug_frequency_shift_data[index_right_hand,channel,:]
# windows_aug_frequency_shift_data_foot = windows_aug_frequency_shift_data[index_foot,channel,:]
# windows_aug_frequency_shift_data_tongue = windows_aug_frequency_shift_data[index_tongue,channel,:]

windows_aug_bandstop_filter_data = aug_bandstop_filter_data  # 生成数据8
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_bandstop_filter_data_left_hand = windows_aug_bandstop_filter_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_gmm_data_right_hand = windows_aug_gmm_data[index_right_hand,channel,:]
# windows_aug_gmm_data_foot = windows_aug_gmm_data[index_foot,channel,:]
# windows_aug_gmm_data_tongue = windows_aug_gmm_data[index_tongue,channel,:]

windows_aug_time_reverse_data = aug_time_reverse_data  # 生成数据9
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_time_reverse_data_left_hand = windows_aug_time_reverse_data[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_gmm_data_right_hand = windows_aug_gmm_data[index_right_hand,channel,:]
# windows_aug_gmm_data_foot = windows_aug_gmm_data[index_foot,channel,:]
# windows_aug_gmm_data_tongue = windows_aug_gmm_data[index_tongue,channel,:]

windows_aug_gmm_data = aug_gmm_data  # 生成数据10
# labels = get_labels(windows)
#************************************* 可视化数据
windows_aug_gmm_data_left_hand = windows_aug_gmm_data[index_left_hand_data,channel,:]    # 第1行第15个trial的22通道数据
# windows_aug_gmm_data_right_hand = windows_aug_gmm_data[index_right_hand,channel,:]
# windows_aug_gmm_data_foot = windows_aug_gmm_data[index_foot,channel,:]
# windows_aug_gmm_data_tongue = windows_aug_gmm_data[index_tongue,channel,:]

t_start, t_stop = 0, 751
sfreq = 100
def plot_signal(data, ax, t_start, t_stop, alpha, c, label, ls):
    ax.plot(data[t_start:t_stop], alpha=alpha, c=c, label=label, linestyle=ls)

# CC_left_hand = np.corrcoef(window_test_left_hand, windows_aug_noise_data_left_hand)
# CC_left_hand = CC_left_hand[0,1]
# CC_right_hand = np.corrcoef(window_test_right_hand, windows_aug_noise_data_right_hand)
# CC_right_hand = CC_right_hand[0,1]
# CC_foot = np.corrcoef(window_test_foot, windows_aug_noise_data_foot)
# CC_foot = CC_foot[0,1]
# CC_tongue = np.corrcoef(window_test_tongue, windows_aug_noise_data_tongue)
# CC_tongue = CC_tongue[0,1]
# print('CC_left_hand =',CC_left_hand)
# print('CC_right_hand =',CC_right_hand)
# print('CC_foot =',CC_foot)
# print('CC_tongue =',CC_tongue)
# 假设 plot_signal 函数已经定义


# 数据对和相关系数计算
# 数据对和相关系数计算
data_pairs = [
    # (window_test_left_hand, windows_aug_noise_data_left_hand, 'Original_data', 'aug_noise_data'),
    # (window_test_left_hand, windows_aug_sign_flip_data_left_hand, 'Original_data', 'aug_sign_flip_data'),
    # (window_test_left_hand, windows_aug_time_reverse_data_left_hand, 'Original_data', 'aug_time_reverse_data'),
    # (window_test_left_hand, windows_aug_smooth_time_mask_data_left_hand, 'Original_data', 'aug_smooth_time_mask_data'),

    # (window_test_left_hand, windows_aug_ft_surrogate_data_left_hand, 'Original_data', 'aug_ft_surrogate_data'),
    # (window_test_left_hand, windows_aug_frequency_shift_data_left_hand, 'Original_data', 'aug_frequency_shift_data'),
    # (window_test_left_hand, windows_aug_bandstop_filter_data_left_hand, 'Original_data', 'aug_bandstop_data'),
    #
    (window_test_left_hand, windows_aug_channels_shuffle_data_left_hand, 'Original_data', 'aug_channels_shuffle_data'),
    (window_test_left_hand, windows_aug_channels_symmetry_data_left_hand, 'Original_data', 'aug_channels_symmetry_data'),

    (window_test_left_hand, windows_aug_gmm_data_left_hand, 'Original_data', 'aug_gmm_data')
]

CC1 = np.corrcoef(window_test_left_hand, windows_aug_noise_data_left_hand)[0, 1]
CC2 = np.corrcoef(window_test_left_hand, windows_aug_sign_flip_data_left_hand)[0, 1]
CC3 = np.corrcoef(window_test_left_hand, windows_aug_time_reverse_data_left_hand)[0, 1]
CC4 = np.corrcoef(window_test_left_hand, windows_aug_smooth_time_mask_data_left_hand)[0, 1]


CC5 = np.corrcoef(window_test_left_hand, windows_aug_ft_surrogate_data_left_hand)[0, 1]
CC6 = np.corrcoef(window_test_left_hand, windows_aug_frequency_shift_data_left_hand)[0, 1]
CC7 = np.corrcoef(window_test_left_hand, windows_aug_bandstop_filter_data_left_hand)[0, 1]


CC8 = np.corrcoef(window_test_left_hand, windows_aug_channels_shuffle_data_left_hand)[0, 1]
CC9 = np.corrcoef(window_test_left_hand, windows_aug_channels_symmetry_data_left_hand)[0, 1]
CC10 = np.corrcoef(window_test_left_hand, windows_aug_gmm_data_left_hand)[0, 1]
ccs = [CC8, CC9, CC10]



FONTSIZE = 10
fig, axes = plt.subplots(nrows=len(data_pairs), ncols=1,figsize=(6, 6),sharex=True, sharey=True)

for i, (original, generated, orig_label, gen_label) in enumerate(data_pairs):
    plot_signal(
        original,
        ax=axes[i],
        t_start=t_start,
        t_stop=t_stop,
        alpha=1.0 if i == 0 else 0.8,
        c='k',
        label=orig_label,
        ls='-'
    )
    plot_signal(
        generated,
        ax=axes[i],
        t_start=t_start,
        t_stop=t_stop,
        alpha=1.0 if i == 0 else 0.8,
        c='tab:red',
        label=gen_label,
        ls='-'
    )
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].margins(x=0)

    axes[i].legend(fontsize=FONTSIZE-2, ncol=1, loc='upper right', bbox_to_anchor=(1, 1.2), frameon=True)
    axes[i].text(0.05, 1.00, f'Corr: {ccs[i]:.2f}', transform=axes[i].transAxes, fontsize=FONTSIZE-2, verticalalignment='top')

    # xticks = np.arange(0, 8)  # 生成从 t_start 到 t_stop 的整数序列
    # axes[i].set_xticks(xticks)
    # axes[i].set_xticklabels([str(tick) for tick in xticks])  # 将刻度值转换为字符串


axes[0].set_title(f'{trial}th trial {channel+1} channel', fontsize=FONTSIZE)  # 调整标题字体大小
axes[len(data_pairs)-1].set_xlabel('Samples point', fontsize=FONTSIZE)

plt.subplots_adjust(hspace=1, wspace=1)  # 调整子图之间的间距
fig.tight_layout()
plt.show()
# fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
# fig_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(fig_dir / "FTSurrogate_K.pdf")
# plt.savefig(fig_dir / "FTSurrogate_K.png")
