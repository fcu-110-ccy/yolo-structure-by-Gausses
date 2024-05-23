import cv2
import numpy as np
import matplotlib.pyplot as plt

def spp_layer(input_img):
    # 获取输入图像的尺寸
    h, w = input_img.shape[:2]

    # 定义不同尺寸的池化窗口
    window_sizes = [(h // 13, w // 13), (h // 9, w // 9), (h // 5, w // 5), (1, 1)]

    # 存储每个池化结果的列表
    pools = []

    for size in window_sizes:
        # 应用最大池化
        pool = cv2.resize(cv2.dilate(input_img, np.ones(size, dtype=np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST)
        pools.append(pool)

    # 将所有池化结果沿着深度方向串联
    spp_output = np.concatenate(pools, axis=2)  # Concatenate along the channel axis

    return spp_output

# 读取图像
f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

img = f4

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 应用SPP层
spp_result = spp_layer(img)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(122)
plt.title('SPP Result')
print(spp_result.shape)
# 只显示第一个通道的结果，因为展示所有通道不直观
plt.imshow(spp_result[:, :,0:12:int(12/3)])
plt.axis('off')

plt.show()
