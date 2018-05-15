# pip3 install opencv-contrib-python

import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

def is_skin_color(pix, centers):
    x = pix[0]
    y = pix[1]
    norm0 = np.linalg.norm(tuple(map(operator.sub, pix[:2], centers[0])))
    norm1 = np.linalg.norm(tuple(map(operator.sub, pix[:2], centers[1])))
    if norm0 < norm1:
        return 0
    else:
        return 255

bw_files = [f for f in listdir("SkinDataset/GT_bw/") if isfile(join("SkinDataset/GT_bw/", f))]
gt_files = [f for f in listdir("SkinDataset/GT/") if isfile(join("SkinDataset/GT/", f))]
bw_files = [f for f in bw_files if f != "243.jpg" and f != "278.jpg" and f != ".DS_Store"]
gt_files = [f for f in gt_files if f != "243.jpg" and f != "278.jpg" if f != ".DS_Store"]

if bw_files == None or bw_files == []:
    for f in gt_files:
        im_gray = cv2.imread("SkinDataset/GT/" + f, 0)
        (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
        cv2.imwrite("SkinDataset/GT_bw/" + f, im_bw)

skin_pix = []
no_skin_pix = []

sum_skin_u = 0
sum_skin_v = 0
sum_no_skin_u = 0
sum_no_skin_v = 0

im_bw = None
quant = 0
for f in gt_files:
    im_color = cv2.imread("SkinDataset/ORI/" + f, 1)
    im_bw = cv2.imread("SkinDataset/GT_bw/" + f, 0)
    im_yuv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
    # if quant == 3:
    #     break;
    for i in range(0, im_yuv.shape[0]):
        for j in range(0, im_yuv.shape[1]):
            # print(im_yuv[i, j], im_yuv[i, j][1:])
            if im_bw[i, j] != 0:
                skin_pix.append(im_yuv[i, j][1:])  # 2 LAST DIMENSIONS
                sum_skin_u = im_yuv[i, j][0]
                sum_skin_v = im_yuv[i, j][1]
            else:
                no_skin_pix.append(im_yuv[i, j][1:])  # 2 LAST DIMENSIONS
                sum_no_skin_u = im_yuv[i, j][0]
                sum_no_skin_v = im_yuv[i, j][1]
            # print(skin_pix)
    quant += 1
# print(skin_pix)
skin_pix = np.array([[f[0], f[1]] for f in skin_pix])
no_skin_pix = np.array([[f[0], f[1]] for f in no_skin_pix])
# print(skin_pix)
heatmap_skin, xedges, yedges = np.histogram2d(skin_pix[:,0], skin_pix[:,1], bins=50)
extent_skin = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

heatmap_no_skin, xedges, yedges = np.histogram2d(no_skin_pix[:,0], no_skin_pix[:,1], bins=50)
extent_no_skin = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure('Skin')
plt.imshow(heatmap_skin.T, origin='lower')
plt.figure('No Skin')
plt.imshow(heatmap_no_skin.T, origin='lower')
heat_diff = np.multiply(heatmap_skin[:,:], heatmap_no_skin[:,:])
plt.figure('DIFF')
plt.imshow(heat_diff, origin='lower')
plt.colorbar()
plt.show()
# skin = plt.scatter(*zip(*skin_pix), color='r', s=3, alpha=0.01, marker='.', label='Skin')
# no_skin = plt.scatter(*zip(*no_skin_pix), color='b', s=3, alpha=0.01, marker='.', label='No Skin')
# legend = plt.legend()
# for lh in legend.legendHandles:
#     lh.set_alpha(1)
# plt.show()
