import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join

bw_files = [f for f in listdir("SkinDataset/GT_bw/") if isfile(join("SkinDataset/GT_bw/", f))]
gt_files = [f for f in listdir("SkinDataset/GT/") if isfile(join("SkinDataset/GT/", f))]
bw_files = [f for f in bw_files if f != "243.jpg" and f != "278.jpg" and f != ".DS_Store"]
gt_files = [f for f in gt_files if f != "243.jpg" and f != "278.jpg" if f != ".DS_Store"]

if bw_files == None or bw_files == []:
    for f in gt_files:
        im_gray = cv2.imread("SkinDataset/GT/" + f, 0)
        (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
        cv2.imwrite("SkinDataset/GT_bw/" + f, im_bw)

svm_vector = []
svm_results = []
block_size = 2
max_block_pos = 1 + (2 * block_size)

print("Generating Svm Vector ...")

for f in gt_files:
    im_color = cv2.imread("SkinDataset/ORI/" + f, 1)
    im_bw = cv2.imread("SkinDataset/GT_bw/" + f, 0)
    im_yuv = cv2.cvtColor(im_color, cv2.COLOR_BGR2LUV)

    for i in range(block_size, im_yuv.shape[0]-max_block_pos):
        for j in range(block_size, im_yuv.shape[1]-max_block_pos):
            patch = im_yuv[i:i+max_block_pos, j:j+max_block_pos]
            patch_vector = []

            for k in range(0, max_block_pos):
                for l in range(0, max_block_pos):
                    patch_vector.append(patch[k, l][0])
                    patch_vector.append(patch[k, l][1])
                    patch_vector.append(patch[k, l][2])

            svm_vector.append(patch_vector)
            svm_results.append(im_bw[i+block_size][j+block_size])

    print("  ", f)

print("Svm Vector Generated !!!")

svm = None
if not isfile('svm_data.dat'):
    print("generating svm .dat")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_POLY)
    svm.setDegree(2)
    svm.setCoef0(1)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(np.float32(svm_vector), cv2.ml.ROW_SAMPLE, np.array(svm_results, np.int32))
    svm.save('svm_data.dat')
else:
    svm = cv2.ml.SVM_load('svm_data.dat')

im243 = cv2.imread("SkinDataset/ORI/243.jpg", 1)
im278 = cv2.imread("SkinDataset/ORI/278.jpg", 1)

im243_yuv = cv2.cvtColor(im243, cv2.COLOR_BGR2LUV)
im278_yuv = cv2.cvtColor(im278, cv2.COLOR_BGR2LUV)

im243_bw = im_bw.copy()
im278_bw = im_bw.copy()

print("Generating Svm Result Images ...")

# img 243
for i in range(block_size, im243_yuv.shape[0]-max_block_pos):
    for j in range(block_size, im243_yuv.shape[1]-max_block_pos):
        patch = im243_yuv[i:i+max_block_pos, j:j+max_block_pos]
        patch_vector = []

        for k in range(0, max_block_pos):
            for l in range(0, max_block_pos):
                patch_vector.append(patch[k, l][0])
                patch_vector.append(patch[k, l][1])
                patch_vector.append(patch[k, l][2])

        a, b = svm.predict(np.array(patch_vector, np.float32).reshape(-1, 3 * (max_block_pos**2)))
        im243_bw[i+block_size, j+block_size] = int(b[0, 0])

# img 278
for i in range(block_size, im278_yuv.shape[0]-max_block_pos):
    for j in range(block_size, im278_yuv.shape[1]-max_block_pos):
        patch = im278_yuv[i:i+max_block_pos, j:j+max_block_pos]
        patch_vector = []

        for k in range(0, max_block_pos):
            for l in range(0, max_block_pos):
                patch_vector.append(patch[k, l][0])
                patch_vector.append(patch[k, l][1])
                patch_vector.append(patch[k, l][2])

        a, b = svm.predict(np.array(patch_vector, np.float32).reshape(-1, 3 * (max_block_pos**2)))
        im278_bw[i+block_size, j+block_size] = int(b[0, 0])

cv2.imwrite("SkinDataset/GT_bw_results_svm/243.jpg", im243_bw)
cv2.imwrite("SkinDataset/GT_bw_results_svm/278.jpg", im278_bw)

print("Svm Result Images Generated !!!")

# Acurácia & Jaccard Index

correct_pix_count = 0
jaccard_div = 0
im_bw = cv2.imread("SkinDataset/GT_bw/243.jpg", 0)
im_bw_result = cv2.imread("SkinDataset/GT_bw_results_svm/243.jpg", 0)

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
im_bw_result = cv2.imread("SkinDataset/GT_bw_results_svm/278.jpg", 0)

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
