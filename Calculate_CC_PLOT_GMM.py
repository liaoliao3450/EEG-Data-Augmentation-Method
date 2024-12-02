
import numpy as np
from scipy.interpolate import interp1d
from plot_utils import FONTSIZE
from plot_utils import plot_signal
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt


def swap_columns_two(matrix1, matrix2,random_state):
    # 将两个矩阵合并成一个大矩阵，计算相关系数矩阵
    corr_matrix = np.corrcoef(np.vstack((matrix1, matrix2)), rowvar=False)
    # 遍历相关系数矩阵的上三角部分
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            # 如果相关系数大于0.8，则交换两列
            if abs(corr_matrix[i, j]) > random_state:
                if i < matrix1.shape[1]:
                    matrix1[:, [i, j-matrix1.shape[1]]] = matrix1[:, [j-matrix1.shape[1], i]]
                else:
                    matrix2[:, [i-matrix1.shape[1], j]] = matrix2[:, [j, i-matrix1.shape[1]]]
                print("交换列 {} 和 {}，相关系数为 {:.2f}".format(i, j, corr_matrix[i, j]))
                break
    return matrix1, matrix2

def swap_columns(matrix, random_state):
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    # 遍历相关系数矩阵的上三角部分
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            # 如果相关系数大于random_state，则交换两列
            if abs(corr_matrix[i, j]) > random_state:
                matrix[:, [i, j]] = matrix[:, [j, i]]
                print("交换列 {} 和 {}，相关系数为 {:.2f}".format(i, j, corr_matrix[i, j]))
                break
    return matrix
def gaussian_mixture_model(data, n_components):
    """
    对每个trial进行高斯混合模型分解
    :param data: 数据，格式为(trial, channel, n_sample)，这里假设数据已经是 float32 类型
    :param n_components: 高斯混合模型的组件数量
    :return: 分解后的高斯混合模型参数
    """
    trial_num, channel_num, sample_num = data.shape
    gmm_params = []

    for j in range(trial_num):
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.fit(data[j].astype(np.float32))
        gmm_params.append(gmm)

    return gmm_params
def normalize_gfp(X):
    # 计算输入矩阵X每列的标准差
    d = np.std(X, axis=0, ddof=1)

    # 对输入矩阵X进行归一化处理
    X_gfp = X / d
    return X_gfp
def denormalize_gfp(normalized_values, original_min, original_max):
    range_val = original_max - original_min
    return normalized_values * range_val + original_min

data = np.load(f'./data/original_data_1.npy')
original_labels = np.load(f'./data/original_labels_1.npy')
#
final_data = np.load(f'./data/data_generate_sampel_11.npy')
labels_generate = np.load(f'./data/original_labels_13.npy')

windows = data   # 原始数据

# labels = get_labels(windows)
N3_indices_left_hand = np.where(original_labels == 1)  # left_hand
N3_indices_right_hand = np.where(original_labels == 2)  # right_hand
N3_indices_foot = np.where(original_labels == 3)  # foot
N3_indices_tongue = np.where(original_labels == 4)  # tongue
channel = 2
# %%  索引
trial = 15
index_left_hand = N3_indices_left_hand[0][trial]  # left_hand总共为4*18，第1行第15个trial（总第54个trial）
index_right_hand = N3_indices_right_hand[0][trial ]
index_foot = N3_indices_foot[0][trial ]
index_tongue = N3_indices_tongue[0][trial ]
#************************************* 可视化数据
# window_test_left_hand = windows[index_left_hand,channel,:]    # 第1行第15个trial的22通道数据
# window_test_right_hand = windows[index_right_hand,channel,:]
# window_test_foot = windows[index_foot,channel,:]
# window_test_tongue = windows[index_tongue,channel,:]

window_test_psd_left_hand = windows[N3_indices_left_hand,channel,:]  # left_hand第2通道数据
window_test_psd_right_hand = windows[N3_indices_right_hand,channel,:]  # right_hand第2通道数据
window_test_psd_foot = windows[N3_indices_foot,channel,:]  # foot第2通道数据
window_test_psd_tongue = windows[N3_indices_tongue,channel,:]  # tongue第2通道数据


window_test_left_hand = windows[N3_indices_left_hand]    # 第1行第15个trial的22通道数据 54072*22
window_test_right_hand = windows[N3_indices_right_hand]
window_test_foot = windows[N3_indices_foot]
window_test_tongue = windows[N3_indices_tongue]

trial_num = window_test_left_hand.shape[0]
channel_num = window_test_left_hand.shape[1]
n_samples = window_test_left_hand.shape[2]


# 沿着第二个维度（channel）拼接数据
X = []
for trial in range(trial_num):
    temp = window_test_left_hand[trial,:,:]
    temp = np.reshape(temp, (channel_num, n_samples))  # D*N
    temp = temp.T  # N*D

    if len(X) == 0:
        X = temp
    else:
        X = np.concatenate((X, temp), axis=0)
print(X.shape)  # 输出：(54072, 22)

###  归一化数据
# X = normalize_gfp(X)
# mean_x = np.mean(X, axis=0)
# std_x = np.std(X, axis=0, ddof=1)
# X = (X - mean_x) / std_x

n_components = 11# 设置高斯混合模型的组件数量(8,10,12)就出错
gmm_1 = GaussianMixture(n_components=n_components, covariance_type='tied') # spherical  diag  full
gmm_1.fit(X.astype(np.float32))
data_generate = gmm_1.sample(54072)[0]

channel_num = 22
n_samples = 751  # 设置生成数据的样本数量

means = gmm_1.means_  # 10*22
covariances = gmm_1.covariances_  # 10*22
weitghs = gmm_1.weights_   # 10
probs = gmm_1.predict_proba(X)  # 54072*10
n_size = 30

# data_generate_sampel = np.zeros((trial_num, channel_num, n_samples))

fitted_values = np.zeros((n_components, channel_num)) # 10*22
    # # # 遍历每个数据点
for j in range(channel_num):
    # 遍历每个簇
    for m in range(gmm_1.n_components):
        # 计算数据点j属于簇i的概率
        # prob = probs[j, i].astype(np.float32)
        # 计算数据点j属于簇i的拟合值
        fitted_values[m] = np.random.multivariate_normal(mean=means[m], cov=np.diagflat(covariances[m]))

# weights = gmm_1.weights_  #[:, np.newaxis]    转换为(10, 1)的形状
# weights = gmm_1.weights_.reshape(-1, 1)
# 计算加权拟合值
weighted_probs_values = gmm_1.weights_ * probs
# weighted_probs_values = swap_columns(weighted_probs_values, 0.7)
weighted_probs_values = normalize_gfp(weighted_probs_values)
original_min = np.min(X)
original_max = np.max(X)
weighted_probs_values = denormalize_gfp(weighted_probs_values, original_min, original_max)


# fitted_values = normalize_gfp(fitted_values)
# original_min = np.min(X)
# original_max = np.max(X)
# fitted_values = denormalize_gfp(fitted_values, original_min, original_max)

data_generate_sampel = np.matmul(weighted_probs_values ,fitted_values)
# data_generate_sampel = normalize_gfp(data_generate_sampel)
# original_min = np.min(X)
# original_max = np.max(X)
# data_generate_sampel = denormalize_gfp(data_generate_sampel, original_min, original_max)/10

# np.save(f'./data/data_generate_sampel10.npy', data_generate_sampel)
# data_generate_sampel = data_generate_sampel * std_x + mean_x

window_test_left_hand = window_test_left_hand
data_generate = data_generate.reshape(72, 751, 22)
data_generate_sampel = data_generate_sampel.reshape(72, 751, 22)
data_generate_sampel = np.transpose(data_generate_sampel, (0, 2, 1))
data_generate = np.transpose(data_generate, (0, 2, 1))

window_test_left_hand0 = window_test_left_hand[2, 5, :]
data_generate = data_generate[2, 5, :]
data_generate_sampel0 = data_generate_sampel[2, 5, :]

window_test_left_hand1 = window_test_left_hand[2, 10, :]
data_generate_sampel1 = data_generate_sampel[2, 10, :]

CC_data = np.corrcoef(data_generate, window_test_left_hand0)
CC_data1 = np.corrcoef(data_generate_sampel0, window_test_left_hand0)
CC_fitted_values = np.corrcoef(data_generate, data_generate_sampel0)
CC_fitted_values = CC_fitted_values[0,1]
CC_data1 = CC_data1[0,1]
CC_data = CC_data[0,1]
print('CC_data = ',CC_data)
print('CC_data1 = ',CC_data1)
print('CC_fitted_values = ',CC_fitted_values)
# print('generated_data = ',generated_data)


# 创建插值函数
x = np.arange(len(window_test_left_hand0))
f_window_test_left_hand = interp1d(x, window_test_left_hand0, kind='linear')
f_data_generate = interp1d(x, data_generate, kind='linear')
f_data_generate_sampel = interp1d(x, data_generate_sampel0, kind='linear')
f_window_test_left_hand1 = interp1d(x, window_test_left_hand1, kind='linear')
f_data_generate_sampel1 = interp1d(x, data_generate_sampel1, kind='linear')


# 生成新的平滑数据
new_x = np.linspace(0, len(window_test_left_hand0) - 1, num=1000)  # 生成更多的点以获得更平滑的效果
smoothed_window_test_left_hand = f_window_test_left_hand(new_x)
smoothed_data_generate = f_data_generate(new_x)
smoothed_data_generate_sampel = f_data_generate_sampel(new_x)
smoothed_data_generate_sampel1 = f_data_generate_sampel1(new_x)
smoothed_window_test_left_hand1 = f_window_test_left_hand1(new_x)



t_start, t_stop = 0, 751
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
t_start, t_stop = 0, 751
sfreq = 100

plot_signal(
    smoothed_window_test_left_hand,
    ax=axes[0],
    t_start=t_start,
    t_stop=t_stop,
    alpha=1.0,
    c='k',
    label='Original left_hand data',
)
plot_signal(
    smoothed_data_generate_sampel,
    ax=axes[0],
    t_start=t_start,
    t_stop=t_stop,
    alpha=1,
    c='tab:red',
    linestyle='-',
    label='Generate left_hand data',
)
plot_signal(
    smoothed_data_generate_sampel1,
    ax=axes[1],
    t_start=t_start,
    t_stop=t_stop,
    alpha=1,
    c='tab:red',
    linestyle='-',
    label='Generate left_hand data',
)
plot_signal(
    smoothed_window_test_left_hand1,
    ax=axes[1],
    t_start=t_start,
    t_stop=t_stop,
    alpha=1.0,
    c='k',
    label='Original left_hand data',
)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticklabels([])
    ax.margins(x=0)
    ax.set_title(f'trial channel')
axes[0].legend(fontsize=FONTSIZE, ncol=2, loc='upper right',
               bbox_to_anchor=(1, 1.1), frameon=True)
axes[1].legend(fontsize=FONTSIZE, ncol=2, loc='upper right',
               bbox_to_anchor=(1, 1.1), frameon=True)

#                bbox_to_anchor=(1, 1.1), frameon=True)

axes[1].set_xlabel('Time (s)', fontsize=FONTSIZE)
fig.tight_layout()
# fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
# fig_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(fig_dir / "FTSurrogate_K.pdf")
# plt.savefig(fig_dir / "FTSurrogate_K.png")
plt.show()

# 13.单个trial按照N_SAMPLEA聚类生成的数据存储
# np.save(f'./data/data_generate_sampel0.npy', data_generate_sampel)