# pip3 install opencv-contrib-python

import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

skin_pix = set()
no_skin_pix = set()

sum_skin_u = 0
sum_skin_v = 0
sum_no_skin_u = 0
sum_no_skin_v = 0

im_bw = None
j = 0
for f in gt_files:
    im_color = cv2.imread("SkinDataset/ORI/" + f, 1)
    im_bw = cv2.imread("SkinDataset/GT_bw/" + f, 0)
    im_yuv = cv2.cvtColor(im_color, cv2.COLOR_BGR2YCR_CB)
    if j > 3:
        break
    else:
        j +=1
        for i in range(0, im_yuv.shape[0]):
            for j in range(0, im_yuv.shape[1]):
                if im_bw[i, j] != 0:
                    skin_pix.add((im_yuv[i, j][0], im_yuv[i, j][1], im_yuv[i, j][2]))
                else:
                    no_skin_pix.add((im_yuv[i, j][0], im_yuv[i, j][1], im_yuv[i, j][2]))

# skin = plt.scatter(*zip(*skin_pix), color='r', alpha=0.01, marker='.', label='Skin')
# no_skin = plt.scatter(*zip(*no_skin_pix), color='b', alpha=0.01, marker='.', label='No Skin')
# legend = plt.legend()
# for lh in legend.legendHandles:
#     lh.set_alpha(1)
# plt.show()


# Print clusteres plotted data
# fig1 = plt.figure()
# ax_clstrd = Axes3D(fig1)
# ax_clstrd.set_xlabel("Y")
# ax_clstrd.set_ylabel("Cr")
# ax_clstrd.set_zlabel("Cb")
# ax_clstrd.scatter(*zip(*skin_pix), color='r', s=1, label='Skin', depthshade=True)
# ax_clstrd.scatter(*zip(*no_skin_pix), color='b', s=1,  label='No Skin', depthshade=True)

skin = plt.scatter(*zip(*skin_pix), color='r', s=1, alpha=0.01, marker='.', label='Skin')
no_skin = plt.scatter(*zip(*no_skin_pix), color='b', s=1, alpha=0.01, marker='.', label='No Skin')
legend = plt.legend()
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.show()
