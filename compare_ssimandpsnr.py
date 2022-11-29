import os
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)  #2

img_ori_path = 'E:/Zhu/ETreg/paper/compareimg/synapse1206/myown/ori/'
img_fine_path = 'E:/Zhu/ETreg/paper/compareimg/synapse1206/myown/fine/'
img_final_path = 'E:/Zhu/ETreg/paper/compareimg/synapse1206/myown/final/'

img_ori_lists = os.listdir(img_ori_path)
img_fine_lists = os.listdir(img_fine_path)
img_Final_lists = os.listdir(img_final_path)
Oriimg_ssim = []
Finimg_ssim = []
Finalimg_ssim = []
xindex = []

for i in range(0, len(img_fine_lists)-1, 2):
    xindex.append(i / 2)

    image1 = cv.imread(img_ori_path + img_ori_lists[i])
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

    image2 = cv.imread(img_ori_path + img_ori_lists[i + 1])
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    sim = psnr(image1, image2)
    Oriimg_ssim.append(sim)

    image1 = cv.imread(img_fine_path + img_fine_lists[i])
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

    image2 = cv.imread(img_fine_path + img_fine_lists[i + 1])
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    sim = psnr(image1, image2)
    Finimg_ssim.append(sim)

    image1 = cv.imread(img_final_path + img_Final_lists[i])
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)  #

    image2 = cv.imread(img_final_path + img_Final_lists[i + 1])
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)  #

    sim = psnr(image1, image2)
    Finalimg_ssim.append(sim)


print(np.mean(Oriimg_ssim))
print(Oriimg_ssim[0])
print(np.mean(Finimg_ssim))
print(Finimg_ssim[0])
print(np.mean(Finalimg_ssim))
print(Finalimg_ssim[0])

y1 = np.array([1] * len(Finimg_ssim))

y2 = Oriimg_ssim
y3 = np.array([np.mean(Oriimg_ssim)] * len(Finimg_ssim))

y4 = Finimg_ssim
y5 = np.array([np.mean(Finimg_ssim)] * len(Finimg_ssim))

y6 = Finalimg_ssim
y7 = np.array([np.mean(Finalimg_ssim)] * len(Finimg_ssim))


x1 = xindex
x2 = np.random.uniform(0, len(Finimg_ssim), len(Finimg_ssim))


fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)

ax1.set_title('PSNR comparison of adjacent images between adjacent volumes')

ax1.set_xlabel('Index-value of adjacent volumes')

ax1.set_ylabel('PSNR-value')
#
# ax1.scatter(x1, y2, s=50, c='c')
# ax1.scatter(x1, y4, s=50, c='m')
ax1.scatter(x1, y4, s=50, c='c')
ax1.scatter(x1, y6, s=50, c='m')

# ax1.plot(x2, y1, c='w', label="Ideal value")
# ax1.plot(x2, y3, c='c', ls='dashed', label="Before alignment")
# ax1.plot(x2, y5, c='m', ls='dashed', label="After alignment")
ax1.plot(x2, y5, c='c', ls='dashed', label="Before restore")
ax1.plot(x2, y7, c='m', ls='dashed', label="After restore")

# plt.xlim(xmax=len(Finimg_ssim)-1+0.5, xmin=0)
# plt.ylim(ymax=1.1, ymin=0)

plt.legend()  # 显示图例
plt.show()


