import cv2
import numpy as np


def histogram_equalization(image):
    h, w = image.shape
    n = h * w
    # 计算直方图
    histogram = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            histogram[image[i, j]] += 1

    # 计算累积分布函数（CDF） (累加每个灰度值的像素值)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    # 归一化累积分布函数
    cdf_normalized = (cdf - cdf.min()) * 255 / (n - cdf.min())

    # 计算新灰度值
    new_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            new_image[i, j] = cdf_normalized[image[i, j]]

    return new_image


def logarithmic_enhance(image, param):
    """
    param 一般为：255/(log(1+max)) max是图像中最大的像素点
    :param image:
    :param param:
    :return:
    """
    h, w = image.shape
    new_image = np.zeros_like(image)

    # img转为float类型
    image = image.astype(float)
    for i in range(h):
        for j in range(w):
            new_image[i, j] = param * np.log(1 + image[i, j])

    # 归一化
    max_hist = new_image.max()
    min_hist = new_image.min()
    for i in range(h):
        for j in range(w):
            new_image[i, j] = (new_image[i, j] - min_hist) / (max_hist - min_hist) * 255
    return new_image


def powerlaw_enhance(image, param, c=1):
    """
    param > 1 增强暗区
    param < 1 增强亮区
    :param image:
    :param param:
    :param c: 默认为1
    :return:
    """
    h, w = image.shape
    new_image = np.zeros_like(image)
    # 转换为float
    image = image.astype(float)

    try:
        # 遍历图像，计算新的像素值
        for i in range(h):
            for j in range(w):
                new_image[i, j] = (image[i, j] ** param) * c
    except Exception as e:
        print(e)
        print('param:', param, 'c:', c, 'image_max:', image.max(), 'image_min:', image.min())

    # 归一化
    max_hist = new_image.max()
    min_hist = new_image.min()
    for i in range(h):
        for j in range(w):
            new_image[i, j] = (new_image[i, j] - min_hist) / (max_hist - min_hist) * 255

    return new_image

