
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

import matplotlib.cm as cm
from deep_learning.csp import FBCSP
from deep_learning.decompositionbase import generate_filterbank



def plt_tsne(data, label, per):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    data = data.cpu().detach().numpy()
    # 对每个trial的所有channels进行平均
    data = np.transpose(data, (0, 2, 1))
    data = np.mean(data, axis=1)
    if isinstance(label, np.ndarray):
        label = torch.tensor(label)
    label = label.cpu().detach().numpy()

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm, label

# ,'aug_bandstop_filter'
# 'aug_bandstop_filter','aug_noise', 'aug_smooth_time_mask','aug_channels_symmetry',
#  'aug_sign_flip', 'aug_ft_surrogate','aug_channels_shuffle', 'aug_frequency_shift',
j=0
aug_method = ["aug_gmm_transform","aug_time_reverse","aug_bandstop_filter","aug_noise","aug_smooth_time_mask",
           "aug_channels_symmetry","aug_sign_flip","aug_ft_surrogate","aug_channels_shuffle",'aug_frequency_shift']
aug_method = aug_method[j]

i = 0
final_data = np.load(f'../data/{aug_method}_data_{i + 1}.npy')
final_data = final_data.astype(np.float64)
labels_generate = np.load(f'../data/{aug_method}_labels_{i + 1}.npy')
labels_generate = labels_generate - 1

data = np.load(f'../data/original_data_{i + 1}.npy')
data = data.astype(np.float64)
labels = np.load(f'../data/original_labels_{i + 1}.npy')
labels = labels - 1

final_data = np.concatenate((final_data, data), axis=0)
labels_generate = np.concatenate((labels_generate, labels), axis=0)

wp=[(4,8),(8,12),(12,30)]
ws=[(2,10),(6,14),(10,32)]

filterbank = generate_filterbank(wp,ws,srate=250,order=4,rp=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(dataset, original_labels, test_size=0.2, random_state=42)

# 初始化 MultiCSP 对象并进行训练
FBCSP = FBCSP(n_components=4, multiclass='ovr',filterbank=filterbank)
FBCSP.fit(data, labels)

# 转换训练集和测试集的数据为特征
X_train_features = FBCSP.transform(data)
X_test_features = FBCSP.transform(final_data)

# # 使用简单的分类器进行分类（例如线性 SVM）
# classifier = SVC()
# classifier.fit(X_train_features, y_train)
# y_pred = classifier.predict(X_test_features)


# 困惑度参数
perplexity = 20

# 获取t-SNE结果
X_norm_original, labels_original = plt_tsne(X_train_features, labels, perplexity)
X_norm_final, labels_final = plt_tsne(X_test_features, labels_generate, perplexity)

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# 绘制第一个子图
for i in range(X_norm_original.shape[0]):
    axs[0].scatter(X_norm_original[i, 0], X_norm_original[i, 1], color=cm.Set1(labels_original[i]))
    axs[0].text(X_norm_original[i, 0], X_norm_original[i, 1], str(labels_original[i]), fontsize=9, ha='right')
axs[0].set_title('Original Data t-SNE')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].grid(True)

# 绘制第二个子图
for i in range(X_norm_final.shape[0]):
    axs[1].scatter(X_norm_final[i, 0], X_norm_final[i, 1], color=cm.Set1(labels_final[i]))
    axs[1].text(X_norm_final[i, 0], X_norm_final[i, 1], str(labels_final[i]), fontsize=9, ha='right')
axs[1].set_title('Augmented Data t-SNE')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].grid(True)

plt.show()