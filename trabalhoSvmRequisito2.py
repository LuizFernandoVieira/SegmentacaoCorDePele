import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join

def find_between(s, first, last):
    start = s.index(first) + len(first)
    end = s.index(last, start)
    return s[start:end]

bw_files = [f for f in listdir("sfa/GT_bw/") if isfile(join("sfa/GT_bw/", f))]
gt_files = [f for f in listdir("sfa/GT/") if isfile(join("sfa/GT/", f))]
bw_files = [f for f in bw_files if f != ".DS_Store"]
gt_files = [f for f in gt_files if f != ".DS_Store"]

if bw_files == None or bw_files == []:
    for f in gt_files:
        im_gray = cv2.imread("sfa/GT/" + f, 0)
        (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
        cv2.imwrite("sfa/GT_bw/" + f, im_bw)

bw_files = [f for f in listdir("sfa/GT_bw_results_svm/") if isfile(join("sfa/GT_bw_results_svm/", f))]
bw_files = [f for f in bw_files if f != ".DS_Store" and int(find_between(f, "(", ")")) > 782]
gt_files = [f for f in gt_files if int(find_between(f, "(", ")")) <= 782]

gt_files_int = [int(find_between(x, "(", ")")) for x in gt_files]
gt_files_int.sort()
gt_files = [("img (" + str(f) + ").jpg") for f in gt_files_int]

block_size = 1
max_block_pos = 1 + (2 * block_size)

if bw_files == None or bw_files == []:

    print("Generating Svm Vector ...")

    svm_vector = []
    svm_results = []

    if not isfile('sfa/svm_data.dat'):

        for f in gt_files:
            print(f)

            svm_vector_it_skin = []
            svm_vector_it_no_skin = []
            svm_results_it = []

            im_color = cv2.imread("sfa/ORI/" + f, 1)
            im_bw = cv2.imread("sfa/GT_bw/" + f, 0)
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

                    if im_bw[i+block_size][j+block_size] == 255:
                        svm_vector_it_skin.append(patch_vector)
                    else:
                        svm_vector_it_no_skin.append(patch_vector)

            svm_vector_it_skin = np.array(svm_vector_it_skin)
            svm_vector_it_skin_avg = np.mean(svm_vector_it_skin, axis=1)

            svm_vector_it_no_skin = np.array(svm_vector_it_no_skin)
            svm_vector_it_no_skin_avg = np.mean(svm_vector_it_no_skin, axis=1)

            svm_vector.append(svm_vector_it_skin_avg)
            svm_vector.append(svm_vector_it_no_skin_avg)

            svm_results.append(255)
            svm_results.append(0)

        print("Svm Vector Generated !!!")

        svm = None

        print("Generating Svm .dat")

        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_POLY)
        svm.setDegree(2)
        svm.setCoef0(1)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(2.67)
        svm.setGamma(5.383)

        sv = [[c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23], c[24], c[25], c[26]] for c in svm_vector]
        sv = np.array(sv, np.float32)
        N = len(sv)
        sv = sv.reshape(N, -1)

        svm.train(np.float32(sv), cv2.ml.ROW_SAMPLE, np.array(svm_results, np.int32))
        svm.save('sfa/svm_data.dat')
    else:
        svm = cv2.ml.SVM_load('sfa/svm_data.dat')

    gt_files = [f for f in listdir("sfa/GT/") if isfile(join("sfa/GT/", f))]
    gt_files = [f for f in gt_files if int(find_between(f, "(", ")")) > 782]
    gt_files_int = [int(find_between(x, "(", ")")) for x in gt_files]
    gt_files_int.sort()
    gt_files = [("img (" + str(f) + ").jpg") for f in gt_files_int]

    for f in gt_files:
        print(f)

        im_bw = cv2.imread("sfa/GT_bw/" + f, 0)
        im = cv2.imread("sfa/ORI/" + f, 1)
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2LUV)

        for i in range(block_size, im_yuv.shape[0]-max_block_pos):
            for j in range(block_size, im_yuv.shape[1]-max_block_pos):
                patch = im_yuv[i:i+max_block_pos, j:j+max_block_pos]
                patch_vector = []

                for k in range(0, max_block_pos):
                    for l in range(0, max_block_pos):
                        patch_vector.append(patch[k, l][0])
                        patch_vector.append(patch[k, l][1])
                        patch_vector.append(patch[k, l][2])

                a, b = svm.predict(np.array(patch_vector, np.float32).reshape(-1, 3 * (max_block_pos**2)))
                im_bw[i+block_size, j+block_size] = int(b[0, 0])

        cv2.imwrite("sfa/GT_bw_results_svm/" + f, im_bw)

    print("Generating Svm Result Images ...")
    print("Svm Result Images Generated !!!")

# Acurácia & Jaccard Index

gt_files = [f for f in listdir("sfa/GT/") if isfile(join("sfa/GT/", f))]
gt_files = [f for f in gt_files if int(find_between(f, "(", ")")) > 782]
gt_files_int = [int(find_between(x, "(", ")")) for x in gt_files]
gt_files_int.sort()
gt_files = [("img (" + str(f) + ").jpg") for f in gt_files_int]

for f in gt_files:
    print(f)

    correct_pix_count = 0
    jaccard_div = 0
    im_bw = cv2.imread("sfa/GT_bw/" + f, 0)
    im_bw_result = cv2.imread("sfa/GT_bw_results_svm/" + f, 0)

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
    print("Acurracy Img: ", acurracy)
    print("Jaccard Index Img: ", jaccard_index)
