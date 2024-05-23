import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_convolution(image, kernel_size):
    # 应用高斯模糊来模拟卷积层
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def simulate_gelan_architecture(image):
    # 模拟GELAN的不同分支
    
    sp1,sp2 = np.array_split(image, 2, axis=1)
    
    conv_small = apply_convolution(sp2, 3)
    conv_medium = apply_convolution(conv_small, 5)
    conv_large = apply_convolution(conv_medium, 7)
    
    # 模拟各层之间的连接
    transition_small_to_image = apply_convolution(image, 5)
    transition_small_to_medium = apply_convolution(conv_small, 5)
    transition_medium_to_large = apply_convolution(conv_medium, 5)
    transition_medium_to_hurge = apply_convolution(conv_large, 5)
    
    sp1 = cv2.resize(sp1,(transition_small_to_image.shape[1],transition_small_to_image.shape[0]))
    transition_medium_to_hurge = cv2.resize(transition_medium_to_hurge,(transition_small_to_image.shape[1],transition_small_to_image.shape[0]))
    transition_medium_to_large = cv2.resize(transition_medium_to_large,(transition_small_to_image.shape[1],transition_small_to_image.shape[0]))
    transition_small_to_medium = cv2.resize(transition_small_to_medium,(transition_small_to_image.shape[1],transition_small_to_image.shape[0]))
    print(sp1.shape)
    print(transition_medium_to_hurge.shape)
    print(transition_medium_to_large.shape)
    print(transition_small_to_image.shape)
    print(transition_small_to_medium.shape)
   # 模拟与原始图像的合并
    merged = np.concatenate((sp1,transition_small_to_image, transition_small_to_medium, transition_medium_to_large,transition_medium_to_hurge), axis=2)
    
    return merged

f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

img = f4
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 模拟GELAN架构
gelan_result = simulate_gelan_architecture(img_rgb)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(122)
plt.title('Simulated GELAN Result')
# 由于结果包含多个通道，只显示前三个通道
print(gelan_result.shape)
plt.imshow(gelan_result[:, :, 0:15:int(15/3)])
plt.axis('off')

plt.show()
