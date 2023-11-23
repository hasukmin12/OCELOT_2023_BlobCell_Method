import os
import cv2
join = os.path.join
import numpy as np
import time

img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/images/StainNorm/train/cell"
seg_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/train/BlobCell_one_label"
# img_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/val/tissue'
# seg_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/tissue'
output_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/blobcell_vis/train/BlobCell_one_label'
os.makedirs(output_path, exist_ok=True)

img_list = sorted(next(os.walk(img_path))[2])
seg_list = sorted(next(os.walk(seg_path))[2])


for case in img_list:
    if case != 'Thumbs.db':
        img = cv2.imread(join(img_path, case))
        seg = cv2.imread(join(seg_path, case))
        print(seg.max())

        t1 = time.time()
        rst = np.zeros(img.shape, dtype=np.uint8)
        for x in range(rst.shape[0]):
            for y in range(rst.shape[1]):
                # print(seg[x][y])
                if seg[x][y][0] == 1: # BGR
                    rst[x][y] = [0, 255, 0]   # BGR  - BC (Green)
                elif seg[x][y][0] == 2:
                    rst[x][y] = [0, 0, 255]   # TC (Red)
                elif seg[x][y][0] == 3:
                    rst[x][y] = [255, 0, 0]  # TC (Red)
                # elif seg[x][y].all() == 3:
                #     rst[x][y] = [0, 255, 0]  # TC (Red)         

        alpha = 0.6
        rst = cv2.addWeighted(img, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)

        t2 = time.time()
        print(f'Colored finished: {case}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
        cv2.imwrite(join(output_path, 'vis_pred' + case), rst)