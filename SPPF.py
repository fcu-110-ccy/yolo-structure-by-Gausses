#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:29:33 2024

@author: york
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sppf_layer(input_img):
    # 获取输入图像的尺寸
    h, w = input_img.shape[:2]

    # 定义池化的尺寸
    pool_size = (5, 5)  # 池化窗口大小为5x5

    # 最大池化操作
    pool1 = cv2.resize(cv2.dilate(input_img, np.ones(pool_size, dtype=np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST)
    pool2 = cv2.resize(cv2.dilate(pool1, np.ones(pool_size, dtype=np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST)
    pool3 = cv2.resize(cv2.dilate(pool2, np.ones(pool_size, dtype=np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST)

    # 将池化结果沿深度方向串联
    sppf_output = np.concatenate([pool1, pool2, pool3,input_img], axis=2)

    return sppf_output

# 读取图像
f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

img = f4

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 应用SPPF层
sppf_result = sppf_layer(img)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(122)
plt.title('SPPF Result')
# 只显示前三个通道的结果，因为展示所有通道不直观
plt.imshow(sppf_result[:, :, 0:12:int(12/3)])
print(sppf_result.shape)
plt.axis('off')

plt.show()
