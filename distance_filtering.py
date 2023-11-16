import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def fitting_pnpresult(tran_pnp, tran_gt):
    # 原始数据
    data = tran_pnp[:, 2].copy()

    # window_size = 100
    # rolling_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    # rolling_std = np.std(np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)]),
    #                      axis=1)
    # padding = len(data) - len(rolling_std)
    # rolling_std = np.pad(rolling_std, (padding // 2, padding - padding // 2), 'constant')
    #
    # coeff = 0.05
    # upper_limit = rolling_mean + coeff * rolling_mean
    # lower_limit = rolling_mean - coeff * rolling_mean

    neg_abnormals = np.where((data <= 0))[0]

    for i in range(len(data)):
        if i in neg_abnormals:
            left, right = i - 1, i + 1
            while left in neg_abnormals and left >= 0:
                left -= 1
            while right in neg_abnormals and right < len(data):
                right += 1
            if left >= 0 and right < len(data):
                data[i] = (data[left] + data[right]) / 2

    outliers = set()
    window_size = 200

    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        q1 = np.percentile(window, 25)  # 第一四分位数
        q3 = np.percentile(window, 75)  # 第三四分位数
        iqr = q3 - q1  # IQR
        lower_bound = q1 - 1.5 * iqr  # 下界
        upper_bound = q3 + 1.5 * iqr  # 上界
        temp_set = set(np.where((data[i:i+window_size] > upper_bound) | (data[i:i+window_size] < lower_bound))[0] + i)
        outliers = outliers.union(temp_set)

    for i in range(len(data)):
        if i in outliers:
            left, right = i - 1, i + 1
            while left in outliers and left >= 0:
                left -= 1
            while right in outliers and right < len(data):
                right += 1
            if left >= 0 and right < len(data):
                data[i] = (data[left] + data[right]) / 2

    print(len(outliers))

    # # 使用窗口大小为window_size进行移动平均滤波
    # window_size = 200
    # smoothed_data = moving_average(data, window_size)
    # # smoothed_data = data

    window_size = 25
    smoothed_data = medfilt(data, kernel_size=window_size)

    return smoothed_data


if __name__ == '__main__':
    sequence_idx = 13
    i = sequence_idx
    dataset = torch.load(os.path.join('data/dataset_work/3DPW/', 'test' + '.pt'))
    name_sequence = dataset['name'][i]

    directory_path = './%s' % name_sequence
    pnp_result = np.load('%s/pnp_distance.npy' % directory_path)
    gt_result = np.load('%s/gt_distance.npy' % directory_path)

    filtered_result = fitting_pnpresult(pnp_result, gt_result)

    # 绘制平滑后的曲线
    plt.figure(figsize=(16, 9))

    plt.scatter(range(len(gt_result)), pnp_result[:, 2], s=10, label='pnp', color='red')
    plt.scatter(range(len(gt_result)), filtered_result, s=10, label='smoothed pnp', color='blue')
    plt.scatter(range(len(gt_result)), gt_result[:, 2], s=10, label='ground truth', color='green')

    # plt.ylim(0, max(gt_result[:, 2]) + 1)
    plt.legend()
    plt.show()

