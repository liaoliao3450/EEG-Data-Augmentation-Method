import os
import numpy as np
import torch
import mne
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

# 定义交换列的函数
def swap_columns(matrix1, matrix2,random_state):
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

# def generate_data(num_samples, means, covariances, weights):
#     num_components = len(means)
#     data = []
#     for _ in range(num_samples):
#         component = np.random.choice(num_components, p=weights)
#         sample = multivariate_normal.rvs(mean=means[component], cov=covariances[component])
#         data.append(sample)
#     return np.array(data)

def min_max_normalize_np(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def min_max_normalize_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def standardization(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma
# ##############################
# #######################

# def gaussian_mixture_model(data):
#     n_trials, n_channels, n_samples = data.shape
#     results = []
#
#     for trial in range(n_trials):
#         trial_results = []
#         for channel in range(n_channels):
#             # 提取（channel，N_sample）矩阵
#             matrix = data[trial, channel, :]
#
#             # 创建并拟合高斯混合模型
#             gmm = GaussianMixture(n_components=20)
#             gmm.fit(matrix.reshape(-1, 1))
#
#             # 将结果添加到trial_results列表中
#             trial_results.append(gmm)
#             # means[trial, channel] = gmm.means_
#             # covariances[trial, channel] = gmm.covariances_
#             # weights[trial, channel] = gmm.weights_
#             # probabilities[trial, channel] = gmm.predict_proba(matrix.reshape(-1, 1))[:, 0]
#
#         # 将trial_results添加到results列表中
#         results.append(trial_results)
#
#     return results

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


def generate_data_from_gmm(gmm_params, channel_num, n_samples):
    """
    根据高斯混合模型参数生成拟合的原始数据
    :param gmm_params: 高斯混合模型参数
    :param n_samples: 生成数据的样本数量
    :return: 生成的数据
    """

    trial_num = len(gmm_params)
    data = np.zeros((trial_num,channel_num, n_samples))
    # data = np.zeros((trial_num, n_samples, channel_num))
    for j in range(trial_num):
        # data[j] = gmm_params[j].sample(n_samples)[0]  # 按行 通道聚类采样
        data[j] = gmm_params[j].sample(channel_num)[0]  # 按列 采样点聚类采样

    return data

subject = ["A01T","A02T","A03T","A04T","A05T","A06T","A07T","A08T","A09T"]
# sub_id = subject[8]
# print(sub_id)
# 1.导入数据
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
i = 0
while i < 9:
    sub_id = subject[i]
    loadpath = "E:\\eeg\\Deepseparator\\BCI-data\\data-gdf"
    filetype = ".gdf"
    filename = os.path.join(loadpath, sub_id + filetype)
    raw = mne.io.read_raw_gdf(filename)
    # events, events_dict = mne.events_from_annotations(raw)
    raw.load_data()
    # 2. 标记坏导
    raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    # 3. 滤波
    raw.filter(4., 32., fir_design='firwin')
    # 4. 全脑平均重参考
    raw_ref = raw.set_eeg_reference(ref_channels='average')
    # 5. ICA
    raw_ica = raw_ref.copy()
    ica = mne.preprocessing.ICA(n_components=22, random_state=97, max_iter=800)
    ica.fit(raw_ica.copy())
    # eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name='EEG-Pz')
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica, ch_name=['EOG-left', 'EOG-central', 'EOG-right'])
    ica.exclude = eog_indices
    #  ica.plot_components()
    #   plt.show()
    # 去除眼动成分
    ica.apply(raw_ica)
    # 6. 删除坏通道
    info = raw_ica.info
    # raw.filter(4., 40., fir_design='firwin')
    # raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    # 7. 提取epochs
    events, event_dict = mne.events_from_annotations(raw_ica)
    tmin, tmax = 1., 4.
    # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
    # Used Annotations descriptions:['1023', '1072', '276', '277', '32766', '768', '769', '770', '771', '772']
    event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
    # event_id = dict({'768': 6})
    epochs = mne.Epochs(raw_ica, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    # 8.得到归一化数据 trial*channel*N_sample
    original_labels = epochs.events[:, -1]- 6
    data = epochs.get_data()*100000

    np.save(f'./data/original_data_{i+1}.npy', data)
    np.save(f'./data/original_labels_{i + 1}.npy', original_labels)
    i += 1
    print(sub_id)
'''
    # 9.设置GMM模型参数
    # data = np.transpose(data, (0, 2, 1))
    n_components = 10  # 设置高斯混合模型的组件数量(8,10,12)就出错
    gmm_params = gaussian_mixture_model(data,n_components)
    channel_num = 22
    n_samples = 751  # 设置生成数据的样本数量

    # 10.基于GMM生成数据
    generated_data = generate_data_from_gmm(gmm_params, channel_num,n_samples)
    final_data = generated_data  ## np.transpose(generated_data, (0, 2, 1))
    generated_labels = original_labels

    # 11.储存数据
    np.save(f'./data/data_generate_{i+1}3.npy', final_data)
    # np.save(f'./data/labels_generate_{i}.npy', generated_labels)

    # 12.单个trial按照N_SAMPLEA聚类分解
    trial_num = data.shape[0]
    channel_num = data.shape[1]
    n_samples = data.shape[2]
    data_generate_sampel = np.zeros((trial_num, channel_num, n_samples))
    for k in range(trial_num):

        data_ = np.transpose(data[k])
        # data = data.transpose((0, 2, 1)).reshape(data.shape[1], -1)  # 转置平展
        # n_components = 10  # 设置高斯混合模型的组件数量(8,10,12)就出错
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag',max_iter=500)
        gmm.fit(data_.astype(np.float32))
        probs = gmm.predict_proba(data_)  # 751*10

        max_values = np.max(probs, axis=0)
        # 初始化计数器
        counts = np.zeros(10, dtype=int)

        # 遍历每一行，找到最大值所在的列并计数
        for row in probs:
            max_index = np.argmax(row)
            counts[max_index] += 1

        print("各列最大值计数结果：", counts)

        # 获取每个簇的均值和协方差矩阵
        means = gmm.means_  # 10*22
        covariances = gmm.covariances_  # 10*22
        weitghs = gmm.weights_   # 10


        fitted_values = np.zeros((n_components, channel_num))
        # # # 遍历每个数据点
        for j in range(channel_num):
            # 遍历每个簇
            for m in range(gmm.n_components):
                # 计算数据点j属于簇i的概率
                # prob = probs[j, i].astype(np.float32)
                # 计算数据点j属于簇i的拟合值
                fitted_values[m]= np.random.multivariate_normal(mean=means[m], cov=np.diag(covariances[m]))

        weights = gmm.weights_[:, np.newaxis]*10  # 转换为(10, 1)的形状
        # 计算加权拟合值
        weighted_fitted_values = weights * fitted_values   # 结果形状为(10, 22)
        # 现在，我们可以将probs与weighted_fitted_values相乘
        data_generate_sampel[k] = np.transpose(np.dot(probs, weighted_fitted_values))  # 结果形状为(751, 22)

    # 13.单个trial按照N_SAMPLEA聚类生成的数据存储
    np.save(f'./data/data_generate_sampel_{i+1}3.npy', data_generate_sampel)

    print(sub_id)

    # 14.下一个被试
    i += 1

    # 15.单个trial按照N_SAMPLEA聚类生成的数据比较相关系数
    # CC_data = np.corrcoef(data[0], data_generate_sampel[0])
    # CC_data = CC_data[0, 1]
    # print('CC_data =', CC_data)
    # plt.plot(data_generate_sampel[0,1,:], label='data_generate_sampel[0]')
    # plt.plot(data[0,1,:], label='data[0]')
    #
    # plt.xlabel('x轴')
    # plt.ylabel('y轴')
    # # plt.title('data_generate_[0] 和 data_[0] 曲线图')
    # plt.legend()
    #
    # plt.show()
'''