

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_convolution(image, kernel_size, factor):
    # 应用卷积，这里简化为高斯模糊
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)




def simulate_elan_architecture(image):
    # 对原始图像应用不同大小的卷积核
    
    sp1,sp2 = np.array_split(image, 2, axis=1)
    
    conv_x1 = apply_convolution(sp1, 3, 1)  # 小卷积核
    conv_x2 = apply_convolution(conv_x1, 5, 1)  # 中卷积核
    conv_x3 = apply_convolution(conv_x2, 7, 1)  # 大卷积核
    
    
    conv_x3A = apply_convolution(image, 7, 1)  # 大卷积核
    print(sp2.shape)
    print(conv_x1.shape)
    print(conv_x2.shape)
    print(conv_x3.shape)
    
    sp2 = cv2.resize(sp2, (conv_x3A.shape[1],conv_x3A.shape[0]))
    conv_x1 = cv2.resize(conv_x1, (conv_x3A.shape[1],conv_x3A.shape[0]))
    conv_x2 = cv2.resize(conv_x2, (conv_x3A.shape[1],conv_x3A.shape[0]))
    conv_x3 = cv2.resize(conv_x3, (conv_x3A.shape[1],conv_x3A.shape[0]))
    
    
    print(conv_x3A.shape)
    # 沿着最后一个维度（颜色通道）合并结果
    concatenated = np.concatenate((conv_x1, conv_x2, conv_x3,sp2,conv_x3A), axis=2)
    return concatenated

# 读取图像
f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

img = f4
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 模拟ELAN架构
elan_result = simulate_elan_architecture(img_rgb)
print(elan_result.shape)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(122)
plt.title('Simulated ELAN Result')
# 由于结果包含多个通道，只显示前三个通道
plt.imshow(elan_result[:, :,0:15:int(15/3)])
plt.axis('off')

plt.show()
