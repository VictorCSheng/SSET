import cv2
import cc3d
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
import scipy

imgvolume = tiff.imread('E:\\Zhu\\ETreg\\code\\ETcoarse\\lables.tif')
for i in range(imgvolume.shape[0]):
    imgtemp = imgvolume[i,:,:]
    retval, imgtemp = cv2.threshold(imgtemp, 127, 255, cv2.THRESH_BINARY)
    imgvolume[i, :, :] = imgtemp

labels_out, N = cc3d.connected_components(imgvolume, return_N=True)
vesicle_stats = cc3d.statistics(labels_out)

##
centroids = vesicle_stats['centroids'][1:]

prememvolume = tiff.imread('E:\\Zhu\\ETreg\\code\\ETcoarse\\preLabels.tif')
for i in range(prememvolume.shape[0]):
    imgtemp = prememvolume[i,:,:]
    retval, imgtemp = cv2.threshold(imgtemp, 127, 255, cv2.THRESH_BINARY)
    prememvolume[i, :, :] = imgtemp

point2planedis = []
for i in range(N):
    centroidtemp = np.around(centroids[i])
    imgtemp = prememvolume[int(centroidtemp[0]), :, :]
    index = np.argwhere(imgtemp == 255)
    rowarray = np.round(index[:, 0])
    pointy = centroidtemp[1]
    colindex = np.where(rowarray == pointy)
    colarray = np.round(index[:, 1][colindex])
    pointx = colarray.min()
    diftemp = pow(pow(centroidtemp[1] - pointy,2) + pow(centroidtemp[2] - pointx,2), 0.5)
    diftemp = diftemp * 0.664
    point2planedis.append(diftemp)

point2planedis = np.array(point2planedis)
print(np.mean(point2planedis))

## figure 1
# #
# # #
# # plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')
# palette = pyplot.get_cmap('Set1')
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 18,
# }
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# # #
# # ax.set_title('Intra Synaptic vesicle volume frequency histogram')
# n, bins, patches = plt.hist(point2planedis, [0,20,40,60,80,100,120,140,160], density=True, color='steelblue', alpha=0.8, edgecolor='k')
# plt.tick_params(top='off', right='off')
# plt.xticks(bins,bins)
# for num,bin in zip(n,bins):
#     plt.annotate("%.4f"%num,xy=(bin,num),xytext=(bin+0.2,num+0.0005))
# #
# normal_x = np.linspace(point2planedis.min(), point2planedis.max(), 1000)
# normal_y = scipy.stats.norm.pdf(normal_x, point2planedis.mean(), point2planedis.std())
# #
# line1, = plt.plot(normal_x,normal_y,'r-', linewidth = 2)
# #
# line_x = np.array([point2planedis.mean()] * 1000)
# line_y = np.linspace(0, n.max()+0.0006, 1000)
# line2, = plt.plot(line_x,line_y, color ="orange", linestyle='--')
# # line2, = plt.axvline(x = point2planedis.mean(), color ="orange", linestyle ="--")
# #
# plt.legend([line1, line2],['Fitted normal distribution curve', 'Average volume of vesicles'],loc='best')
# #
# ax.set_xlabel('Distance between vesicles in synapses and the presynaptic membrane (*nm)')
# #
# ax.set_ylabel('Frequency')
# plt.show()

## figure 2
##
voxel_counts = vesicle_stats['voxel_counts'][1:]
V_volume = voxel_counts
xindex = list(range(N))
for i in range(N):
    V_volume[i] = voxel_counts[i] * 0.664 * 0.664 * 0.664
print(N)
print(np.mean(V_volume))

ordered_point2planedis_inx = np.argsort(point2planedis)
ordered_point2planedis = point2planedis[ordered_point2planedis_inx]
ordered_V_volume = V_volume[ordered_point2planedis_inx]

V_volume_0_20 = V_volume[np.where(point2planedis <= 20)]
yvalue1 = np.array([np.mean(V_volume_0_20)] * 1000)

tempP = point2planedis[np.where(point2planedis > 20)]
tempV = V_volume[np.where(point2planedis > 20)]
V_volume_20_40 = tempV[np.where(tempP <= 40)]
yvalue2 = np.array([np.mean(V_volume_20_40)] * 1000)

tempP = point2planedis[np.where(point2planedis > 40)]
tempV = V_volume[np.where(point2planedis > 40)]
V_volume_40_60 = tempV[np.where(tempP <= 60)]
yvalue3 = np.array([np.mean(V_volume_40_60)] * 1000)

tempP = point2planedis[np.where(point2planedis > 60)]
tempV = V_volume[np.where(point2planedis > 60)]
V_volume_60_80 = tempV[np.where(tempP <= 80)]
yvalue4 = np.array([np.mean(V_volume_60_80)] * 1000)

tempP = point2planedis[np.where(point2planedis > 80)]
tempV = V_volume[np.where(point2planedis > 80)]
V_volume_80_100 = tempV[np.where(tempP <= 100)]
yvalue5 = np.array([np.mean(V_volume_80_100)] * 1000)

tempP = point2planedis[np.where(point2planedis > 100)]
tempV = V_volume[np.where(point2planedis > 100)]
V_volume_100_120 = tempV[np.where(tempP <= 120)]
yvalue6 = np.array([np.mean(V_volume_100_120)] * 1000)

tempP = point2planedis[np.where(point2planedis > 120)]
tempV = V_volume[np.where(point2planedis > 120)]
V_volume_120_140 = tempV[np.where(tempP <= 140)]
yvalue7 = np.array([np.mean(V_volume_120_140)] * 1000)

tempP = point2planedis[np.where(point2planedis > 140)]
tempV = V_volume[np.where(point2planedis > 140)]
V_volume_140_160 = tempV[np.where(tempP <= 160)]
yvalue8 = np.array([np.mean(V_volume_140_160)] * 1000)

tempP = point2planedis[np.where(point2planedis > 160)]
tempV = V_volume[np.where(point2planedis > 160)]
V_volume_160_180 = tempV[np.where(tempP <= 180)]
yvalue9 = np.array([np.mean(V_volume_160_180)] * 1000)

V_volume_mean = np.array([np.mean(V_volume)] * N)
V_volume_std = np.array([np.std(V_volume)] * N)
r1 = V_volume_mean - V_volume_std
r2 = V_volume_mean + V_volume_std

#
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
fig = plt.figure()
#
ax = fig.add_subplot(1, 1, 1)
x_major_locator=MultipleLocator(20)#
ax.xaxis.set_major_locator(x_major_locator)

plt.xlim(0, 180)
plt.ylim(0, 4300)
#
line_y = np.array([4300] * 1000)
line_x1 = np.linspace(0, 20, 1000)
ax.fill_between(line_x1, 0, line_y, color='ghostwhite', alpha=0.6)
line_x2 = np.linspace(20, 40, 1000)
ax.fill_between(line_x2, 0, line_y, color='lavender', alpha=0.6)
line_x3 = np.linspace(40, 60, 1000)
ax.fill_between(line_x3, 0, line_y, color='lightsteelblue', alpha=0.2)
line_x4 = np.linspace(60, 80, 1000)
ax.fill_between(line_x4, 0, line_y, color='cornflowerblue', alpha=0.2)
line_x5 = np.linspace(80, 100, 1000)
ax.fill_between(line_x5, 0, line_y, color='b', alpha=0.1)
line_x6 = np.linspace(100, 120, 1000)
ax.fill_between(line_x6, 0, line_y, color='cornflowerblue', alpha=0.2)
line_x7 = np.linspace(120, 140, 1000)
ax.fill_between(line_x7, 0, line_y, color='lightsteelblue', alpha=0.2)
line_x8 = np.linspace(140, 160, 1000)
ax.fill_between(line_x8, 0, line_y, color='lavender', alpha=0.6)
line_x9 = np.linspace(160, 180, 1000)
ax.fill_between(line_x9, 0, line_y, color='ghostwhite', alpha=0.6)
#
ax.scatter(ordered_point2planedis, ordered_V_volume, c='c')

# #
xindex1 = np.linspace(0, 20, 1000)
ax.plot(xindex1, yvalue1, color='orange', linewidth=3.0)

xindex2 = np.linspace(20, 40, 1000)
ax.plot(xindex2, yvalue2, color='orange', linewidth=3.0)

xindex3 = np.linspace(40, 60, 1000)
ax.plot(xindex3, yvalue3, color='orange', linewidth=3.0)

xindex4 = np.linspace(60, 80, 1000)
ax.plot(xindex4, yvalue4, color='orange', linewidth=3.0)

xindex5 = np.linspace(80, 100, 1000)
ax.plot(xindex5, yvalue5, color='orange', linewidth=3.0)

xindex6 = np.linspace(100, 120, 1000)
ax.plot(xindex6, yvalue6, color='orange', linewidth=3.0)

xindex7 = np.linspace(120, 140, 1000)
ax.plot(xindex7, yvalue7, color='orange', linewidth=3.0)

xindex8 = np.linspace(140, 160, 1000)
ax.plot(xindex8, yvalue8, color='orange', linewidth=3.0)

xindex9 = np.linspace(160, 180, 1000)
line1, = ax.plot(xindex9, yvalue9, color='orange', linewidth=3.0)

xindex = np.linspace(0, 180, N)
line2, = ax.plot(xindex, V_volume_mean, color='r', linestyle ="--", linewidth=3.0)
#
plt.legend([line1, line2],['Mean value of vesicle volume in each range', 'Mean value of all vesicle'],loc='best')
#
ax.set_xlabel('Distance between vesicles in synapses and the presynaptic membrane (*nm)')
#
ax.set_ylabel('Vesicle volume (*nm^3)')
plt.show()
