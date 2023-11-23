import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
join = os.path.join

import csv


anno_path = '/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/annotations/test/cell'
anno_list = sorted(next(os.walk(anno_path))[2])

aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/cell_14_n'
os.makedirs(aim_path, exist_ok=True)

output_path_for_visualize = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/cell_seg_visulization/test/cell_14_n'
os.makedirs(output_path_for_visualize, exist_ok=True)

img_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/cell'
img_list = sorted(next(os.walk(img_path))[2])
if "Thumbs.db" in img_list:
    img_list.remove("Thumbs.db")

for case in anno_list:
    if case != 'Thumbs.db':
        print(case)
        anno = open(join(anno_path, case), 'r', encoding='utf-8')
        pt = csv.reader(anno)
        seg = np.zeros((1024, 1024), dtype=np.uint8)

        for line in pt:
            # print(line)
            x = int(line[0])
            y = int(line[1])
            cls = int(line[2])

            if cls == 1:
                cv2.circle(seg, (x,y), 14, 1, -1)
            elif cls == 2:
                cv2.circle(seg, (x,y), 14, 2, -1)

        print(seg.max())

        # mmcv.imwrite(seg.astype(np.uint8), join(aim_path, 'cell_{0:03d}.png'.format(anno_list.index(case))))
        cv2.imwrite(join(aim_path, 'cell_{0:03d}.png'.format(anno_list.index(case)+401)), seg.astype(np.uint8)) # +537


        # overlap rst
        image = cv2.imread(join(img_path, img_list[anno_list.index(case)]))
        # rst = np.zeros(image.shape, dtype=np.uint8)
        rst = image
        for x in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                # print(seg[x][y])
                if seg[x][y] == 1: # BGR
                    rst[x][y] = [255, 0, 0]  # BGR  - Background Cell (연두)
                elif seg[x][y] == 2:
                    rst[x][y] = [0, 0, 255]  # Tumor Cell (빨강)

        # alpha = 0.4
        # rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)
        # rst = cv2.addWeighted(image, 0.5, rst, 0.6, 0.0, dtype = cv2.CV_32F)
        # cv2.imwrite(join(output_path_for_visualize, 'cell_{0:03d}.png'.format(anno_list.index(case)+401)), rst)
        num = int(case[:3])-1
        cv2.imwrite(join(output_path_for_visualize, 'cell_{0:03d}.png'.format(num)), rst)
        anno.close()

        
    