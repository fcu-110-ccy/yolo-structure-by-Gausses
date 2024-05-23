import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')
frame = f4
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 分割frame成三个颜色通道
sp1,sp2 = np.array_split(frame, 2, axis=1)

# 模拟C3模块处理一个通道
def process_channel_c3(channel,sp2):
    # 1x1卷积，可以使用3x3卷积模拟，因为1x1卷积在OpenCV中不直接支持
    conv1x1_1 = cv2.GaussianBlur(channel, (1, 1), 0)
    
    # 串行3x3卷积
    conv3x3_1 = cv2.GaussianBlur(conv1x1_1, (3, 3), 0)
    conv3x3_2 = cv2.GaussianBlur(conv3x3_1, (3, 3), 0)
    conv3x3_3 = cv2.GaussianBlur(conv3x3_2, (3, 3), 0)
    
   
    conv1x1_1 = cv2.GaussianBlur(channel, (1, 1), 0)
    # 调整大小以匹配原始图像的大小，以便可以合并回彩色图像
    #concatenated_resized = cv2.resize(concatenated, (channel.shape[1], channel.shape[0]))
    
    return conv3x3_3,conv1x1_1

# 处理每个通道
conv3x3_3,conv1x1_1 = process_channel_c3(sp1,sp2)

print(conv1x1_1.shape)
print(conv3x3_3.shape)
# 将处理过的三个通道重新组合
conv3x3_3=cv2.resize(conv3x3_3, (frame.shape[1],frame.shape[0]))
conv1x1_1 =cv2.resize(conv1x1_1, (frame.shape[1],frame.shape[0]))
merged_processed = cv2.merge([conv3x3_3, conv1x1_1])




plt.subplot(121)
plt.title('Original')
plt.imshow(frame)

plt.subplot(122)
plt.title('Processed with C3 module using concat')
plt.imshow(merged_processed[:,:,0:6:int(6/3)])

plt.show()
