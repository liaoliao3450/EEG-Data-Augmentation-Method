import torch
from torch import Tensor
from typing import Callable, Union, Tuple, Optional
from numpy.random import default_rng
from braindecode.augmentation.transforms import GaussianNoise, time_reverse, SignFlip, FTSurrogate
from typing import List, Tuple, Any
from numbers import Real
from sklearn.utils import check_random_state
import numpy as np
from braindecode.augmentation.transforms import ChannelsShuffle, FrequencyShift, BandstopFilter, SmoothTimeMask, ChannelsSymmetry

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

Batch = List[Tuple[torch.Tensor, int, Any]]
Output = Tuple[torch.Tensor, torch.Tensor]
def time_reverse(X, y):
    return torch.flip(X, [-1]), y

class Transform(torch.nn.Module):
    operation = None
    def __init__(self, probability=1.0, random_state=None):
        super().__init__()
        if self.forward.__func__ is Transform.forward:
            assert callable(self.operation),\
                "operation should be a ``callable``."

        assert isinstance(probability, Real), (
            f"probability should be a ``real``. Got {type(probability)}.")
        assert probability <= 1. and probability >= 0., \
            "probability should be between 0 and 1."
        self._probability = probability
        self.rng = check_random_state(random_state)

    def get_augmentation_params(self, *batch):
        return dict()

    def forward(self, X, y=None, aug_data=None, aug_labels=None, extra_param=None) -> Output:
        X = torch.as_tensor(X).float()
        out_X = X.clone()

        if y is not None:
            y = torch.as_tensor(y)
            out_y = y.clone()
        else:
            out_y = torch.zeros(X.shape[0])

        if aug_data is not None and aug_labels is not None:
            aug_data = torch.as_tensor(aug_data).float()
            aug_labels = torch.as_tensor(aug_labels)
            out_aug_data = aug_data.clone()
            out_aug_labels = aug_labels.clone()
        else:
            out_aug_data = None
            out_aug_labels = None

        if extra_param is not None:
            out_extra_param = extra_param
        else:
            out_extra_param = None

        # Samples a mask setting for each example whether they should stay unchanged or not
        mask = self._get_mask(X.shape[0])
        num_valid = mask.sum().long()

        if num_valid > 0:
            # Uses the mask to define the output
            if out_aug_data is not None and out_aug_labels is not None:
                out_X[mask, ...], tr_y, tr_z, tr_w = self.operation(
                    out_X[mask, ...], out_y[mask], out_aug_data[mask, ...], out_aug_labels[mask],
                    **self.get_augmentation_params(out_X[mask, ...], out_y[mask], out_aug_data[mask, ...],
                                                   out_aug_labels[mask])
                )
            else:
                out_X[mask, ...], tr_y, tr_z, tr_w = self.operation(
                    out_X[mask, ...], out_y[mask],
                    **self.get_augmentation_params(out_X[mask, ...], out_y[mask])
                )

            # Apply the operation defining the Transform to the whole batch
            if type(tr_y) is tuple:
                out_y = tuple(tmp_y[mask] for tmp_y in tr_y)
            else:
                out_y[mask] = tr_y

        if y is not None and aug_labels is not None:
            return out_X, out_y, out_aug_data, out_aug_labels, out_extra_param
        elif y is not None:
            return out_X, out_y, out_extra_param
        elif aug_labels is not None:
            return out_X, out_aug_data, out_aug_labels, out_extra_param
        else:
            return out_X, out_extra_param

    def _get_mask(self, batch_size=None) -> torch.Tensor:
        """Samples whether to apply operation or not over the whole batch
        """
        return torch.as_tensor(
            self.probability > self.rng.uniform(size=batch_size)
        )

    @property
    def probability(self):
        return self._probability

class TimeReverse(Transform):
    operation = staticmethod(time_reverse)

    def __init__(
        self,
        probability,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )


def _pick_channels_randomly(X, p_pikel, random_state=None):
    rng = check_random_state(random_state)
    mask = rng.binomial(1, 1 - p_pikel, size=X.shape[1]).astype(bool)
    return torch.tensor(mask, dtype=torch.float32, device=X.device)


def gmm_feature(X1, y1, X2, y2, p_pikel, random_state=None):
    mask2 = _pick_channels_randomly(X2, p_pikel, random_state=random_state)

    X2_masked = X2 * mask2.unsqueeze(-1)

    batch_size, n_channels, n_times = X1.shape
    rng = check_random_state(random_state)

    # Randomly select one channel for each sample in the batch
    selected_channels = rng.randint(0, n_channels, size=batch_size)

    # Create a mask for the selected channels
    swap_mask = torch.zeros((batch_size, n_channels, n_times), dtype=torch.bool, device=X1.device)

    for i in range(batch_size):
        swap_mask[i, selected_channels[i], :] = True

    # Swap the data at the selected positions
    X1_swapped = X1.clone()
    X2_swapped = X2.clone()
    X1_swapped[swap_mask] = X2[swap_mask]
    X2_swapped[swap_mask] = X1[swap_mask]

    return X1_swapped, y1, X2_swapped, y2


class GMM_FEATURE(Transform):
    operation = staticmethod(gmm_feature)

    def __init__(self, probability, p_pikel=0.1, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.p_pikel = p_pikel

    def get_augmentation_params(self, *batch):
        return {
            "p_pikel": self.p_pikel,
            "random_state": self.rng,
        }

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

# # 加载数据
i=0
data = np.load(f'./data/original_data_{i+1}.npy')
original_labels = np.load(f'./data/original_labels_{i + 1}.npy')
data = torch.tensor(data)
aug_gmm_data = np.load('data/aug_gmm_data_1.npy')
gmm_labels = np.load('data/original_labels_11.npy')
probability =1

noise_transform = GaussianNoise(probability=probability, std=0.5,random_state=42)  # 添加高斯噪声
# time_reverse_transform = time_reverse(probability=probability)      # 时间反转
sign_flip_transform = SignFlip(probability=probability,random_state=21)            # 符号翻转
ft_surrogate_transform = FTSurrogate(probability=probability,random_state=21)      # 傅里叶变换代理
channels_shuffle_transform = ChannelsShuffle(probability=probability,random_state=23)  # 通道随机打乱
frequency_shift_transform = FrequencyShift(probability=probability, sfreq=4.0,random_state=42)  # 频率偏移
bandstop_filter_transform = BandstopFilter(probability=probability,sfreq=100,random_state=23)  # 带阻滤波
smooth_time_mask_transform = SmoothTimeMask(probability=probability, mask_len_samples=100,random_state=21)  # 时间掩码
channels_symmetry_transform = ChannelsSymmetry(probability=probability, ordered_ch_names=[f'Ch{i+1}' for i in range(22)],random_state=42)  # 通道对称增强
gmm_transform = GMM_FEATURE(probability=probability,random_state=42)  # 通道随机打乱


aug_noise_data = noise_transform(data)

aug_reverse = time_reverse(data,original_labels)
aug_reverse_data =aug_reverse[0]
# aug_reverse_labels =aug_reverse[1]
aug_bandstop_filter_data = bandstop_filter_transform(data)
aug_smooth_time_mask_data = smooth_time_mask_transform(data)
aug_channels_symmetry_data = channels_symmetry_transform(data)
aug_sign_flip_data = sign_flip_transform(data)
aug_ft_surrogate_data = ft_surrogate_transform(data)
aug_channels_shuffle_data = channels_shuffle_transform(data)
aug_frequency_shift_data = frequency_shift_transform(data)
aug_gmm_transform_data = gmm_transform(data,original_labels,aug_gmm_data,gmm_labels)
aug_gmm_transform_data = aug_gmm_transform_data[0]

# 数据列表
data_list = [
    (data, "Original Data"),
    # (aug_noise_data, "Noise"),
    # (aug_sign_flip_data, "Sign Flip"),
    # (aug_reverse_data, "time_reverse"),
    # (aug_smooth_time_mask_data, "Smooth Time Mask"),

    # (aug_ft_surrogate_data, "FT Surrogate"),
    # (aug_frequency_shift_data, "Frequency Shift"),
    # (aug_bandstop_filter_data, "Bandstop Filter"),

    # (aug_channels_symmetry_data, "Channels Symmetry"),
    # (aug_channels_shuffle_data, "Channels Shuffle"),
    (aug_gmm_transform_data, "GMM")
]

# 生成假数据作为示例

for i in range(len(data_list)):
    data_list[i] = (data_list[i][0][3].reshape(22, -1), data_list[i][1])

ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
            'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_types = ['eeg'] * 22
sfreq = 250  # Hz


avg_psds = []

freq_ranges = [(0, 4), (4, 8), (8, 13), (13, 30)]

# 创建九排四张图的布局
fig, axs = plt.subplots(5, len(freq_ranges), figsize=(15, 10))


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
        norm_avg_psds = norm.fit_transform(avg_psds.reshape(-1, 1)).flatten()  # 应用归一化对象
        # im, _ = mne.viz.plot_topomap(norm_avg_psds, current_raw.info, axes=axs[j , i], show=False, cmap='bwr')
        im, _ = mne.viz.plot_topomap(avg_psds, current_raw.info, axes=axs[j , i], show=False, cmap='bwr',vmin=0, vmax=0.10)
        cbar = plt.colorbar(im, ax=axs[j , i], fraction=0.046, pad=0.04)
        # cbar.set_label('Normalized PSD', fontsize=12)
        axs[j , i].set_title(f' {low}-{high} Hz', fontsize=12)
        axs[j , i].set_xlabel(f'{label}_topomap', fontsize=12)

plt.tight_layout()
plt.show()