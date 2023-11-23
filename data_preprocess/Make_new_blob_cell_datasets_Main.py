#%%
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
join = os.path.join
import csv


# 이미지에서 contour를 찾고
# 제공되는 좌표가 해당 contour 안에 들어있다면 그 contour 전체를 segmentation map으로 활용한다
# 만약 contour밖에 있다면 원래 하던대로 r=10으로 학습하자


img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/cell"
img_list = sorted(next(os.walk(img_path))[2])

coord_path = "/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/annotations/test/cell"
coord_list = sorted(next(os.walk(coord_path))[2])

if "Thumbs.db" in img_list:
    img_list.remove("Thumbs.db")

aim_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/BlobCell"
os.makedirs(aim_path, exist_ok=True)

vis_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/cell_seg_visulization/test/BlobCell"
os.makedirs(vis_path, exist_ok=True)




for case in img_list:
    print(case)
    # Load the image
    image = cv2.imread(join(img_path, case))
    # seg = cv2.imread(join(label_path, case))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Apply a binary threshold to the image
    threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    anno = open(join(coord_path, coord_list[img_list.index(case)+0]), 'r', encoding='utf-8') # +80
    pt = csv.reader(anno)
    seg = np.zeros_like(grayscale_image, dtype=np.uint8)
    # mask = np.zeros_like(grayscale_image, dtype=np.uint8)

    useful_contour = []
    contour_minmax = []

    # Iterate over the contours
    for contour in contours:

        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # If the area is greater than a certain threshold,
        # then the contour is a blob
        if 2500> area > 150:

            # Draw the contour on the image
            # cv2.drawContours(image, [contour], -1, (0, 255, 0), -1)

            useful_contour.append(contour)

            x_min = 1024
            x_max = contour[0][0][0]
            y_min = 1024
            y_max = contour[0][0][1]
            for coord in contour:
                if coord[0][0]>x_max:
                    x_max = coord[0][0]
                if coord[0][0]<x_min:
                    x_min = coord[0][0]
                if coord[0][1]>y_max:
                    y_max = coord[0][1]
                if coord[0][1]<y_min:
                    y_min = coord[0][1]
            
            contour_minmax.append([x_min, x_max, y_min, y_max])


    # 정답 값이 contour 안에 포함되어 있으면 해당 countour를 segmentation map으로 활용한다.
    proposed_contour = []

    for line in pt:
        # print(line)
        x = int(line[0])
        y = int(line[1])
        cls = int(line[2])

        not_in_contour = False


        for v in contour_minmax:
            if v[0] < x <v[1]:
                if v[2] < y < v[3]:
                    not_in_contour=True

                    if cls == 1:
                        cv2.drawContours(seg, [useful_contour[contour_minmax.index(v)]], -1, 1, -1)
                        break
                    elif cls == 2:
                        cv2.drawContours(seg, [useful_contour[contour_minmax.index(v)]], -1, 2, -1)
                        break
                    # if cls == 1:
                    #     cv2.drawContours(segmentation_map, [useful_contour[contour_minmax.index(v)]], -1, 1, -1)
                    #     break
                    # elif cls == 2:
                    #     cv2.drawContours(segmentation_map, [useful_contour[contour_minmax.index(v)]], -1, 2, -1)
                    #     break


        if not_in_contour==False:
            if cls == 1:
                cv2.circle(seg, (x,y), 14, 1, -1)
            elif cls == 2:
                cv2.circle(seg, (x,y), 14, 2, -1)

    cv2.imwrite(join(aim_path, case), seg)
    # Image.fromarray(seg).save(join(aim_path, case))


     # overlap rst
    rst = image
    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            # print(seg[x][y])
            if seg[x][y] == 1: # BGR
                rst[x][y] = [255, 0, 0]  # BGR  - Background Cell (연두)
            elif seg[x][y] == 2:
                rst[x][y] = [0, 0, 255]  # Tumor Cell (빨강)

    # # image = cv2.imread(join(img_path, img_list[anno_list.index(case)]))
    # rst = np.zeros(image.shape, dtype=np.uint8)
    # for x in range(seg.shape[0]):
    #     for y in range(seg.shape[1]):
    #         # print(seg[x][y])
    #         if seg[x][y] == 1: # BGR
    #             rst[x][y] = [255, 0, 0]  # BGR  - Background Cell (연두)
    #         elif seg[x][y] == 2:
    #             rst[x][y] = [0, 0, 255]  # Tumor Cell (빨강)

    # alpha = 0.55 # 0.4  0.6
    # rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
    # # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)
    # # cv2.imwrite(join(vis_path, 'cell_{0:03d}.png'.format(img_list.index(case)+0)), rst) # + 401  + 537
    cv2.imwrite(join(vis_path, case), rst) # + 401  + 537
   