#!\Anaconda3\envs\py35 python

# -*- coding: utf-8 -*-
#!@Time   : 2018/6/1 15:39
#!@Author : python
#!@File   : .py



from operator import eq
import numpy as np
import pydicom as dicom
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import thin, reconstruction

import matplotlib.pyplot as plt
import cv2


def regiongrowth(orig_image, seed_matrix, threshold):
    """区域生长算法

    # Arguments：
        orig_image：待分割图像
        seed_matrix：种子矩阵
        threshold：生长条件，即待分割图中的像素值与目标像素值之间的可承受的差值
    # Returns
        seg：最终分割后的mask
        num_region：分割区域的个数

    # Example
        regiongrow(orig_image, seed_matrix, 50)

    """
    orig_image = np.array(orig_image, dtype=np.float32)                         # 将图像数值格式转为float

    markers = thin(seed_matrix)                                                 # 将种子所在区域缩小为一个点，后面限定区域时使用
    coordinate_nonzero = markers.nonzero()                                      # 当种子为矩阵时，获取种子点的坐标值
    coordinate_nonzero = np.transpose(coordinate_nonzero)                       # 获取种子点的坐标值的转置
    seed_value = []
    for coordinate_i in coordinate_nonzero:
        seed_value.append(orig_image[coordinate_i[0], coordinate_i[1]])         # 获取种子点的灰度值

    size_x, size_y = orig_image.shape
    seg_image = np.array(np.zeros((size_x, size_y)), dtype=bool)                # 初始化一个bool格式的矩阵储存分割结果

    for i, each_seed_value in enumerate(seed_value):
        all_satisfied = abs(orig_image - each_seed_value) <= threshold          # 所有满足单个种子点阈值条件的像素点
        seg_image = seg_image | all_satisfied                                   # 所有满足多有种子点阈值条件的像素点

    # seg = reconstruction(coordinate_nonzero, seg_image)


    label_seg_image = label(seg_image)                                          # 将所有连通区域打label
    area_region = regionprops(label_seg_image)                                  # 统计被标记的区域的面积分布，返回值为显示区域总数

    seg = np.array(np.zeros((size_x, size_y)), dtype=np.float32)                # 初始化一个float格式的矩阵储存分割结果
    for masker in coordinate_nonzero:                                           # 提取maskers所在的连通区域
        for each_area_region in area_region:
            for coord in each_area_region.coords:
                if sum(eq(coord, masker)) == 2:                                 # 若连通区域中包含masker，则保留此连通区域
                    for idx in each_area_region.coords:
                        seg[idx[0], idx[1]] = 1
                    break

    seg = label(seg)                                                            # 将有效的label进行标记
    num_region = seg.max()                                                      # 有效连通label的个数
    return seg, num_region


if __name__ == "__main__":
    one_slices = dicom.dcmread(r'F:\DataTest\MIM\1545711775478_c91045accece49deb74a3e4dc207f8d6_711764339\1.2.840.113619.2.55.3.604678789.276.1545612737.223\1.2.840.113619.2.55.3.604678789.276.1545612737.229\1545711770173_8adedde67bdb4f27b819b947f3dfd9b6_1509718234.dcm', force=True)
    orig_image = one_slices.pixel_array
    plt.figure()
    plt.imshow(orig_image)
    plt.show()

    orig_image = (orig_image - np.min(orig_image))/(np.max(orig_image) - np.min(orig_image)) * 255
    orig_image = np.array(orig_image, dtype=np.uint8)

    plt.figure()
    plt.imshow(orig_image)
    plt.show()

    dst_image = np.zeros((orig_image.shape[0], orig_image.shape[1]))

    dst_image = cv2.adaptiveThreshold(orig_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 511, 2)

    dst_image = dst_image / 255 * (np.max(orig_image) - np.min(orig_image)) + np.min(orig_image)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.subplot(1, 2, 2)
    plt.imshow(dst_image)
    plt.show()

    size_x, size_y = orig_image.shape
    orig_image = np.array(orig_image, dtype=np.float32)
    orig_image_min = orig_image.min()
    orig_image = orig_image - orig_image_min
    orig_image_max = orig_image.max()
    orig_image = orig_image / orig_image_max * 255
    orig_image = np.array(orig_image, dtype=np.uint8)

    seed_matrix = np.zeros((size_x, size_y))
    # 指定想要分割的像素的坐标，按照dicom的坐标系
    # coordinate1_x = 158
    # coordinate1_y = 260
    # coordinate2_x = 370
    # coordinate2_y = 260

    coordinate1_x = 138
    coordinate1_y = 252
    coordinate2_x = 346
    coordinate2_y = 252

    # coordinate1_x = 226
    # coordinate1_y = 290
    # coordinate2_x = 308
    # coordinate2_y = 290

    # 矩阵按行号、列号，与dicom正好相反
    seed_matrix[coordinate1_y, coordinate1_x] = orig_image[coordinate1_y, coordinate1_x]
    seed_matrix[coordinate2_y, coordinate2_x] = orig_image[coordinate2_y, coordinate2_x]
    seg_image, num_region = regiongrowth(orig_image, seed_matrix, 10)                         # 需指定阈值

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.subplot(1, 2, 2)
    plt.imshow(seg_image)
    plt.show()



