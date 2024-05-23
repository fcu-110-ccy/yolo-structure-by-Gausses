#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:30:50 2024

@author: york
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def simulate_3x3_conv(image):
    # 模拟3x3卷积，这里使用高斯模糊
    return cv2.GaussianBlur(image, (3, 3), 0)

def simulate_1x1_conv(image):
    # 模拟1x1卷积，实际上1x1在这里就是一个全局的均值操作
    kernel = np.ones((1, 1), np.float32) / (1 * 1)
    return cv2.filter2D(image, -1, kernel)


# 读取图像
f1 = cv2.imread('./test/1.png')
f2 = cv2.imread('./test/2.png')
f3 = cv2.imread('./test/3.jpg')

img = f3
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 应用3x3卷积
img_3x3_conv = simulate_3x3_conv(img_rgb)

# 应用1x1卷积
img_1x1_conv = simulate_1x1_conv(img_rgb)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(132)
plt.title('3x3 Conv Simulation')
plt.imshow(img_3x3_conv)
plt.axis('off')

plt.subplot(133)
plt.title('1x1 Conv Simulation')
plt.imshow(img_1x1_conv)
plt.axis('off')

plt.show()
