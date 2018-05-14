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
        print(f)
        im_gray = cv2.imread("sfa/GT/" + f, 0)
        (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
        cv2.imwrite("sfa/GT_bw/" + f, im_bw)

bw_files = [f for f in listdir("sfa/GT_bw_results/") if isfile(join("sfa/GT_bw_results/", f))]
bw_files = [f for f in bw_files if f != ".DS_Store" and int(find_between(f, "(", ")")) > 782]
gt_files = [f for f in gt_files if int(find_between(f, "(", ")")) <= 782]

gt_files_int = [int(find_between(x, "(", ")")) for x in gt_files]
gt_files_int.sort()
gt_files = [("img (" + str(f) + ").jpg") for f in gt_files_int]

criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_PP_CENTERS #USE_INITIAL_LABELS
im_bw = None

if bw_files == None or bw_files == []:

    labels = None
    centers = []

    if not isfile('sfa/labels.npy') or not isfile('sfa/centers.npy'):

        print("Training ...")

        count = 0

        for f in gt_files:
            print(f)

            skin_pix = []
            no_skin_pix = []
            sum_skin_u = 0
            sum_skin_v = 0
            sum_no_skin_u = 0
            sum_no_skin_v = 0
            z = []

            im_color = cv2.imread("sfa/ORI/" + f, 1)
            im_bw = cv2.imread("sfa/GT_bw/" + f, 0)
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

            centers_im = np.zeros((2, 2))
            centers_im[0, 0] = avg_skin_u
            centers_im[0, 1] = avg_skin_v
            centers_im[1, 0] = avg_no_skin_u
            centers_im[1, 1] = avg_no_skin_v

            all_kmeans_colors = np.array(z, np.float32)

            compactness_im, labels_im, centers_im = cv2.kmeans(all_kmeans_colors, 2, None, criteria, 10, flags, centers_im)
            centers.append(centers_im[0])
            centers.append(centers_im[1])

        sum_skin_u = 0
        sum_skin_v = 0
        sum_no_skin_u = 0
        sum_no_skin_v = 0

        index = 0

        for c in centers:
            if index % 2:
                sum_skin_u += c[0]
                sum_skin_v += c[1]
            else:
                sum_no_skin_u += c[0]
                sum_no_skin_v += c[1]
            index += 1

        avg_skin_u = sum_skin_u / (len(centers) / 2)
        avg_skin_v = sum_skin_v / (len(centers) / 2)
        avg_no_skin_u = sum_no_skin_u / (len(centers) / 2)
        avg_no_skin_v =  sum_no_skin_v / (len(centers) / 2)

        centers_kmeans = np.zeros((2, 2))
        centers_kmeans[0, 0] = avg_skin_u
        centers_kmeans[0, 1] = avg_skin_v
        centers_kmeans[1, 0] = avg_no_skin_u
        centers_kmeans[1, 1] = avg_no_skin_v

        cent = [[c[0], c[1]] for c in centers]
        cent = np.array(cent, np.float32)
        N = len(cent)
        cent = cent.reshape(N, -1)

        compactness, labels, centers = cv2.kmeans(cent, 2, None, criteria, 10, flags, centers_kmeans)

        np.save('sfa/labels', labels)
        np.save('sfa/centers', centers)

    else:
        print("Já treinou")

        labels = np.load('sfa/labels.npy')
        centers = np.load('sfa/centers.npy')
        # im_bw = cv2.imread("SkinDataset/GT_bw/11.jpg", 0)

    # im243 = cv2.imread("SkinDataset/ORI/243.jpg", 1)
    # im278 = cv2.imread("SkinDataset/ORI/278.jpg", 1)
    #
    # im243_yuv = cv2.cvtColor(im243, cv2.COLOR_BGR2YUV)
    # im278_yuv = cv2.cvtColor(im278, cv2.COLOR_BGR2YUV)
    #
    # im243_bw = im_bw.copy()
    # im278_bw = im_bw.copy()
    #
    # for i in range(0, im243_yuv.shape[0]):
    #     for j in range(0, im243_yuv.shape[1]):
    #         im243_bw[i, j] = is_skin_color(im243_yuv[i, j], centers)
    #
    # for i in range(0, im278_yuv.shape[0]):
    #     for j in range(0, im278_yuv.shape[1]):
    #         im278_bw[i, j] = is_skin_color(im278_yuv[i, j], centers)
    #
    # cv2.imwrite("SkinDataset/GT_bw_results/243.jpg", im243_bw)
    # cv2.imwrite("SkinDataset/GT_bw_results/278.jpg", im278_bw)

# im_gray = cv2.imread("SkinDataset/GT/243.jpg", 0)
# (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
# cv2.imwrite("SkinDataset/GT_bw/243.jpg", im_bw)
#
# im_gray = cv2.imread("SkinDataset/GT/278.jpg", 0)
# (thresh, im_bw) = cv2.threshold(im_gray, 8, 255, cv2.THRESH_BINARY)
# cv2.imwrite("SkinDataset/GT_bw/278.jpg", im_bw)

# Acurácia & Jaccard Index

# correct_pix_count = 0
# jaccard_div = 0
# im_bw = cv2.imread("SkinDataset/GT_bw/243.jpg", 0)
# im_bw_result = cv2.imread("SkinDataset/GT_bw_results/243.jpg", 0)
#
# for i in range(0, im_bw_result.shape[0]):
#     for j in range(0, im_bw_result.shape[1]):
#         if im_bw_result[i, j] == im_bw[i, j]:
#             correct_pix_count += 1
#             jaccard_div += 1
#         else:
#             jaccard_div += 2
#
# acurracy = correct_pix_count / (im_bw_result.shape[0] * im_bw_result.shape[1])
# jaccard_index = correct_pix_count / jaccard_div
#
# print("-----------------------")
# print("Acurracy Img 243: ", acurracy)
# print("Jaccard Index Img 243: ", jaccard_index)
#
# correct_pix_count = 0
# im_bw = cv2.imread("SkinDataset/GT_bw/278.jpg", 0)
# im_bw_result = cv2.imread("SkinDataset/GT_bw_results/278.jpg", 0)
#
# for i in range(0, im_bw_result.shape[0]):
#     for j in range(0, im_bw_result.shape[1]):
#         if im_bw_result[i, j] == im_bw[i, j]:
#             correct_pix_count += 1
#             jaccard_div += 1
#         else:
#             jaccard_div += 2
#
# acurracy = correct_pix_count / (im_bw_result.shape[0] * im_bw_result.shape[1])
# jaccard_index = correct_pix_count / jaccard_div
#
# print("-----------------------")
# print("Acurracy Img 278: ", acurracy)
# print("Jaccard Index Img 278: ", jaccard_index)
# print("-----------------------")
