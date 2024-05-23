#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:32:59 2024

@author: york
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_convolution(image, kernel_size, stride, padding):
    # 对图像应用卷积操作，这里使用高斯模糊模拟
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def simulate_c2f_architecture(image):
    # 初始卷积操作
    initial_conv = apply_convolution(image, 3, 1, 0)
    
    # 模拟卷积bottleneck操作
    def conv_bottleneck(conv_input):
        a=apply_convolution(conv_input, 3, 3, 0)
        b=apply_convolution(a, 3, 3, 0)
        
        return np.concatenate([conv_input,b],axis=2)
    
    # 应用第一层卷积bottleneck
    bottleneck1 = conv_bottleneck(initial_conv)
    
    # 分割数据，模拟拆分操作
    split1, split2 = np.array_split(bottleneck1, 2, axis=1)
    
    # 对分割后的每一部分应用卷积bottleneck
    processed_split1 = conv_bottleneck(split1)
    processed_split2 = conv_bottleneck(processed_split1)
    
    # 拼接处理过的分支
    print(image.shape)
    print(bottleneck1.shape)
    print(processed_split1.shape)
    print(processed_split2.shape)
    processed_split1 = cv2.resize(processed_split1, (( image.shape[1], image.shape[0])))
    processed_split2 = cv2.resize(processed_split2, (( image.shape[1],image.shape[0])))

    concatenated = np.concatenate((image,bottleneck1,processed_split1, processed_split2), axis=2)
    
    # 应用最后一层卷积
    final_conv = apply_convolution(concatenated, 3, 1, 0)
    
    return final_conv

# 读取图像

f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')


img = f4
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 模拟C2F架构
c2f_result = simulate_c2f_architecture(img_rgb)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(122)
plt.title('Simulated C2F Result')
plt.imshow(c2f_result[:, :, 0:45:int(45/3)])

plt.axis('off')

plt.show()
