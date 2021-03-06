import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join

def is_skin_color(pix, centers):
    x = pix[0]
    y = pix[1]
    norm0 = np.linalg.norm(tuple(map(operator.sub, pix[1:], centers[0])))
    norm1 = np.linalg.norm(tuple(map(operator.sub, pix[1:], centers[1])))
    if norm0 < norm1:
        return 0
    else:
        return 255

bw_files = [f for f in listdir("SkinDataset/GT_bw/") if isfile(join("SkinDataset/GT_bw/", f))]
gt_files = [f for f in listdir("SkinDataset/GT/") if isfile(join("SkinDataset/GT/", f))]
bw_files = [f for f in bw_files if f != "243.jpg" and f != "278.jpg" and f != ".DS_Store"]
gt_files = [f for f in gt_files if f != "243.jpg" and f != "278.jpg" and f != ".DS_Store"]

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
z = []

if not isfile('SkinDataset/GT_bw_results/243.jpg') or not isfile('SkinDataset/GT_bw_results/278.jpg'):

    labels = None
    centers = None

    if not isfile('SkinDataset/labels.npy') or not isfile('SkinDataset/centers.npy'):
        for f in gt_files:
            im_color = cv2.imread("SkinDataset/ORI/" + f, 1)
            im_bw = cv2.imread("SkinDataset/GT_bw/" + f, 0)
            im_yuv = cv2.cvtColor(im_color, cv2.COLOR_BGR2YUV)

            for i in range(0, im_yuv.shape[0]):
                for j in range(0, im_yuv.shape[1]):
                    if im_bw[i, j] != 0:
                        skin_pix.append(im_yuv[i, j][1:])
                        sum_skin_u = im_yuv[i, j][1]
                        sum_skin_v = im_yuv[i, j][2]
                    else:
                        no_skin_pix.append(im_yuv[i, j][1:])
                        sum_no_skin_u = im_yuv[i, j][1]
                        sum_no_skin_v = im_yuv[i, j][2]
                    z.append((im_yuv[i, j][1], im_yuv[i, j][2]))

        avg_skin_u = sum_skin_u / len(skin_pix)
        avg_skin_v = sum_skin_v / len(skin_pix)
        avg_no_skin_u = sum_no_skin_u / len(no_skin_pix)
        avg_no_skin_v =  sum_no_skin_v / len(no_skin_pix)

        centers = np.zeros((2, 2))
        centers[0, 0] = avg_skin_u
        centers[0, 1] = avg_skin_v
        centers[1, 0] = avg_no_skin_u
        centers[1, 1] = avg_no_skin_v

        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_USE_INITIAL_LABELS
        compactness, labels, centers = cv2.kmeans(np.array(z, np.float32), 2, None, criteria, 10, flags, centers)

        np.save('SkinDataset/labels', labels)
        np.save('SkinDataset/centers', centers)

    else:
        labels = np.load('SkinDataset/labels.npy')
        centers = np.load('SkinDataset/centers.npy')
        im_bw = cv2.imread("SkinDataset/GT_bw/11.jpg", 0)

    im243 = cv2.imread("SkinDataset/ORI/243.jpg", 1)
    im278 = cv2.imread("SkinDataset/ORI/278.jpg", 1)

    im243_yuv = cv2.cvtColor(im243, cv2.COLOR_BGR2YUV)
    im278_yuv = cv2.cvtColor(im278, cv2.COLOR_BGR2YUV)

    im243_bw = im_bw.copy()
    im278_bw = im_bw.copy()

    for i in range(0, im243_yuv.shape[0]):
        for j in range(0, im243_yuv.shape[1]):
            im243_bw[i, j] = is_skin_color(im243_yuv[i, j], centers)

    for i in range(0, im278_yuv.shape[0]):
        for j in range(0, im278_yuv.shape[1]):
            im278_bw[i, j] = is_skin_color(im278_yuv[i, j], centers)

    cv2.imwrite("SkinDataset/GT_bw_results/243.jpg", im243_bw)
    cv2.imwrite("SkinDataset/GT_bw_results/278.jpg", im278_bw)

im_gray = cv2.imread("SkinDataset/GT/243.jpg", 0)
(thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
cv2.imwrite("SkinDataset/GT_bw/243.jpg", im_bw)

im_gray = cv2.imread("SkinDataset/GT/278.jpg", 0)
(thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
cv2.imwrite("SkinDataset/GT_bw/278.jpg", im_bw)

# Acurácia & Jaccard Index

correct_pix_count = 0
jaccard_div = 0
im_bw = cv2.imread("SkinDataset/GT_bw/243.jpg", 0)
im_bw_result = cv2.imread("SkinDataset/GT_bw_results/243.jpg", 0)

for i in range(0, im_bw_result.shape[0]):
    for j in range(0, im_bw_result.shape[1]):
        if im_bw_result[i, j] == im_bw[i, j]:
            correct_pix_count += 1
            jaccard_div += 1
        else:
            jaccard_div += 2

acurracy = correct_pix_count / (im_bw_result.shape[0] * im_bw_result.shape[1])
jaccard_index = correct_pix_count / jaccard_div

print("-----------------------")
print("Acurracy Img 243: ", acurracy)
print("Jaccard Index Img 243: ", jaccard_index)

correct_pix_count = 0
im_bw = cv2.imread("SkinDataset/GT_bw/278.jpg", 0)
im_bw_result = cv2.imread("SkinDataset/GT_bw_results/278.jpg", 0)

for i in range(0, im_bw_result.shape[0]):
    for j in range(0, im_bw_result.shape[1]):
        if im_bw_result[i, j] == im_bw[i, j]:
            correct_pix_count += 1
            jaccard_div += 1
        else:
            jaccard_div += 2

acurracy = correct_pix_count / (im_bw_result.shape[0] * im_bw_result.shape[1])
jaccard_index = correct_pix_count / jaccard_div

print("-----------------------")
print("Acurracy Img 278: ", acurracy)
print("Jaccard Index Img 278: ", jaccard_index)
print("-----------------------")
