
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure

f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')


img = f4
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

def apply_convolution(image, kernel_size, stride, padding):
    # 对图像应用卷积操作，这里使用高斯模糊模拟
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


a = apply_convolution(img, 3, 1, 0)
b = apply_convolution(img, 1, 1, 0)

c = cv2.normalize(img, None,0,100,cv2.NORM_MINMAX)


cat = np.concatenate( (a, b,c),axis = 2)



# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(122)
plt.title('Simulated REP Result')
plt.imshow(cat[:, :, 0:9:int(9/3)])

plt.axis('off')

plt.show()

