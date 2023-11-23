import os
import cv2
join = os.path.join
import numpy as np
import time
from PIL import Image

seg_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/tissue_crop'
aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/tissue_crop2'
os.makedirs(aim_path, exist_ok=True)

seg_list = sorted(next(os.walk(seg_path))[2])


for case in seg_list:
    if case != 'Thumbs.db':
        seg = cv2.imread(join(seg_path, case))
        print()
        print(case)
        print(seg.max())

        t1 = time.time()
        rst = np.zeros(seg.shape, dtype=np.uint8)
        for x in range(rst.shape[0]):
            for y in range(rst.shape[1]):
                # print(seg[x][y])
                if seg[x][y][0] == 0: # BGR
                    rst[x][y] = 1   # BGR  - BC (Green)
                elif seg[x][y][0] == 1:
                    rst[x][y] = 2  # TC (Red)
                elif seg[x][y][0] == 2:
                    rst[x][y] = 3  # TC (Red)

        label_Gray = cv2.cvtColor(rst, cv2.COLOR_RGB2GRAY)
        Image.fromarray(label_Gray).save(join(aim_path, case))