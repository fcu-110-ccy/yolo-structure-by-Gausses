import cv2
import matplotlib.pyplot as plt
import numpy as np

f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')
frame = f4
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

sp1,sp2 = np.array_split(frame, 2, axis=1)
    

# 对每个通道分别进行三次高斯模糊并正规化
def process_channel(channel):
    output1 = cv2.GaussianBlur(channel, (3, 3), 0)
    output2 = cv2.GaussianBlur(output1, (3, 3), 0)
    output3 = cv2.GaussianBlur(output2, (3, 3), 0)
    return cv2.normalize(output3, None, 0, 255, cv2.NORM_MINMAX)

b_processed = process_channel(sp1)

# 对frame进行5x5的高斯模糊
frame_blurred = cv2.GaussianBlur(sp2, (5, 5), 0)

# 将处理过的三个通道重新组合
merged_processed = cv2.merge([b_processed, frame_blurred])

concat_img=merged_processed
# 将原始图像和串联后的图像转换为RGB颜色空间，以便在Matplotlib中正确显示

concat_img = cv2.resize(concat_img,(frame.shape[1],frame.shape[0]))
plt.subplot(121)
plt.title('Original')
plt.imshow(frame)

plt.subplot(122)
plt.title('Concatenated')
plt.imshow(concat_img[:,:,0:6:int(6/3)])

plt.show()
