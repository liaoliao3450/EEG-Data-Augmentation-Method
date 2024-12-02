import torch
import numpy as np
from braindecode.augmentation.transforms import GaussianNoise, time_reverse, SignFlip, FTSurrogate
from braindecode.augmentation.transforms import ChannelsShuffle, FrequencyShift, BandstopFilter, SmoothTimeMask, ChannelsSymmetry
from sklearn.utils import check_random_state
from numbers import Real
from typing import List, Tuple, Any

Batch = List[Tuple[torch.Tensor, int, Any]]
Output = Tuple[torch.Tensor, torch.Tensor]
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

# 定义数据增强的概率
probability = 0.5

# 每个增强实例会以给定概率应用于输入数据
noise_transform = GaussianNoise(probability=probability, std=0.16)  # 添加高斯噪声
# time_reverse_transform = time_reverse()      # 时间反转

sign_flip_transform = SignFlip(probability=probability)            # 符号翻转
ft_surrogate_transform = FTSurrogate(probability=probability)      # 傅里叶变换代理
channels_shuffle_transform = ChannelsShuffle(probability=probability)  # 通道随机打乱
frequency_shift_transform = FrequencyShift(probability=probability, sfreq=2.0)  # 频率偏移
bandstop_filter_transform = BandstopFilter(probability=probability,sfreq=100,random_state=21)  # 带阻滤波
smooth_time_mask_transform = SmoothTimeMask(probability=probability, mask_len_samples=100)  # 时间掩码
channels_symmetry_transform = ChannelsSymmetry(probability=probability, ordered_ch_names=[f'Ch{i+1}' for i in range(22)])  # 通道对称增强
gmm_transform = GMM_FEATURE(probability=probability,random_state=42)  # 通道随机打乱

# 将所有增强方法应用于 EEG 数据
# augmented_data = X.copy()
i = 0
while i < 9:

    data = np.load(f'./data/original_data_{i+1}.npy')
    original_labels = np.load(f'./data/original_labels_{i + 1}.npy')

    aug_gmm_data = np.load(f'./data/aug_gmm_data_{i+1}.npy')
    gmm_labels = np.load(f'./data/aug_gmm_labels_{i + 1}.npy')

    aug_noise = noise_transform(data)
    data = torch.as_tensor(data).float()
    aug_time_reverse= time_reverse(data,original_labels)
    aug_time_reverse_data = aug_time_reverse[0]
    aug_sign_flip = sign_flip_transform(data)
    aug_ft_surrogate = ft_surrogate_transform(data)
    aug_channels_shuffle = channels_shuffle_transform(data)
    aug_frequency_shift = frequency_shift_transform(data)
    aug_bandstop_filter_data = bandstop_filter_transform(data)
    aug_smooth_time_mask = smooth_time_mask_transform(data)
    aug_channels_symmetry = channels_symmetry_transform(data)

    aug_gmm_transform_data = gmm_transform(data, original_labels, aug_gmm_data, gmm_labels)
    aug_gmm_transform_data = aug_gmm_transform_data[0]
    aug_gmm_transform_labels = aug_gmm_transform_data[2]

    # np.save(f'./data/aug_noise_data_{i+1}.npy', aug_noise)
    # np.save(f'./data/aug_noise_labels_{i + 1}.npy', original_labels)

    # np.save(f'./data/aug_time_reverse_data_{i+1}.npy', aug_time_reverse_data)
    # np.save(f'./data/aug_time_reverse_labels_{i + 1}.npy', original_labels)

    # np.save(f'./data/aug_sign_flip_data_{i+1}.npy', aug_sign_flip)
    # np.save(f'./data/aug_sign_flip_labels_{i + 1}.npy', original_labels)
    #
    # np.save(f'./data/aug_ft_surrogate_data_{i+1}.npy', aug_ft_surrogate)
    # np.save(f'./data/aug_ft_surrogate_labels_{i + 1}.npy', original_labels)
    #
    # np.save(f'./data/aug_channels_shuffle_data_{i+1}.npy', aug_channels_shuffle)
    # np.save(f'./data/aug_channels_shuffle_labels_{i + 1}.npy', original_labels)
    #
    # np.save(f'./data/aug_frequency_shift_data_{i+1}.npy', aug_frequency_shift)
    # np.save(f'./data/aug_frequency_shift_labels_{i + 1}.npy', original_labels)

    # np.save(f'./data/aug_bandstop_filter_data_{i+1}.npy', aug_bandstop_filter_data)
    # np.save(f'./data/aug_bandstop_filter_labels_{i + 1}.npy', original_labels)
    #
    # np.save(f'./data/aug_smooth_time_mask_data_{i+1}.npy', aug_smooth_time_mask)
    # np.save(f'./data/aug_smooth_time_mask_labels_{i + 1}.npy', original_labels)
    #
    # np.save(f'./data/aug_channels_symmetry_data_{i+1}.npy', aug_channels_symmetry)
    # np.save(f'./data/aug_channels_symmetry_labels_{i + 1}.npy', original_labels)

    np.save(f'./data/aug_gmm_transform_data_{i+1}.npy', aug_gmm_transform_data)
    np.save(f'./data/aug_gmm_transform_labels_{i + 1}.npy', original_labels)

    i += 1
    print(i)





