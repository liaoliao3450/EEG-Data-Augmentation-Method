
import numpy as np
from sklearn.mixture import GaussianMixture

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
def generate_data(data_, trial_num, channel_num, n_samples):
    X = []
    for trial in range(trial_num):
        temp = data_[trial,:,:]
        temp = np.reshape(temp, (channel_num, n_samples))  # D*N
        temp = temp.T  # N*D

        if len(X) == 0:
            X = temp
        else:
            X = np.concatenate((X, temp), axis=0)

    n_components = 11
    gmm_1 = GaussianMixture(n_components=n_components, covariance_type='tied')
    gmm_1.fit(X.astype(np.float32))

    means = gmm_1.means_  # 10*22
    covariances = gmm_1.covariances_  # 10*22
    # weights = gmm_1.weights_   # 10
    probs = gmm_1.predict_proba(X)  # 54072*10

    data_generate_sample = np.zeros((trial_num,n_samples, channel_num))
    fitted_values = np.zeros((n_components, channel_num)) # 10*22

    for j in range(channel_num):
        for m in range(gmm_1.n_components):
            fitted_values[m] = np.random.multivariate_normal(mean=means[m], cov=np.diagflat(covariances[m]))

    weighted_probs_values = gmm_1.weights_ * probs
    weighted_probs_values = normalize_gfp(weighted_probs_values)
    weighted_probs_values = swap_columns(weighted_probs_values,0.5)
    data_generate_sampel = np.matmul(weighted_probs_values ,fitted_values)/6
    data_generate_sampel = data_generate_sampel.reshape(72, 751, 22)
    data_generate_sampel = np.transpose(data_generate_sampel, (0, 2, 1))

    return data_generate_sampel

i = 0
while i < 9:

    data = np.load(f'./data/original_data_{i+1}.npy')
    original_labels = np.load(f'./data/original_labels_{i + 1}.npy')

    windows = data   # 原始数据
    # labels = get_labels(windows)
    N3_indices_left_hand = np.where(original_labels == 1)  # left_hand
    N3_indices_right_hand = np.where(original_labels == 2)  # right_hand
    N3_indices_foot = np.where(original_labels == 3)  # foot
    N3_indices_tongue = np.where(original_labels == 4)  # tongue

    window_test_left_hand = windows[N3_indices_left_hand]    # 第1行第15个trial的22通道数据 54072*22
    window_test_right_hand = windows[N3_indices_right_hand]
    window_test_foot = windows[N3_indices_foot]
    window_test_tongue = windows[N3_indices_tongue]

    trial_num = window_test_left_hand.shape[0]
    channel_num = window_test_left_hand.shape[1]
    n_samples = window_test_left_hand.shape[2]

    window_test_left_hand_smple = generate_data(window_test_left_hand, trial_num, channel_num, n_samples)
    # print(window_test_left_hand_smple.shape)
    window_test_right_hand_smple = generate_data(window_test_right_hand, trial_num, channel_num, n_samples)
    # print(window_test_right_hand_smple.shape)
    window_test_foot_smple = generate_data(window_test_foot, trial_num, channel_num, n_samples)
    window_test_tongue_smple = generate_data(window_test_tongue, trial_num, channel_num, n_samples)


    data_generate_sample = np.concatenate((
        window_test_left_hand_smple,
        window_test_right_hand_smple,
        window_test_foot_smple,
        window_test_tongue_smple
    ))
    # print(data_generate_sample11.shape)
    indices_ = np.concatenate((N3_indices_left_hand, N3_indices_right_hand, N3_indices_foot, N3_indices_tongue), axis=1)
    indices_flat = indices_.flatten()
    sorted_indices = np.argsort(indices_)
    data_generate_sample = data_generate_sample[sorted_indices]
    # indices_flat = indices_.flatten()
    # data_generate_sample11 = data_generate_sample11[indices_flat]
    data_generate_sample = np.squeeze(data_generate_sample, axis=0)
    # print(data_generate_sample11.shape)
    # np.save(f'./data/data_generate_sampel6.npy', data_generate_sample11)

    np.save(f'./data/aug_gmm_data_{i+1}.npy', data_generate_sample)
    np.save(f'./data/aug_gmm_labels_{i + 1}.npy', original_labels)

    i += 1
    print(i)
