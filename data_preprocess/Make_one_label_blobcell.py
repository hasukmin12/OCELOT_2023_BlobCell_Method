import os
import numpy as np
join = os.path.join
import cv2
from PIL import Image

path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/BlobCell'
aim_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/BlobCell_one_label'
os.makedirs(aim_path, exist_ok=True)

path_list = sorted(next(os.walk(path))[2])

for case in path_list:
    print(case)
    # label = cv2.imread(join(path, case))
    # label = cv2.cvtColor(cv2.imread(join(path, case)), cv2.COLOR_BGR2RGB)
    label = cv2.imread(join(path, case))
    rst = np.zeros(label.shape, dtype=np.uint8)
    # print(label.max())
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            if label[x][y][0] != 0: # BGR
                label[x][y] = [1,1,1]

            # if label[x][y] == 2: # BGR
            #     rst[x][y] = [255, 0, 0]   # BGR  - BC (Green)
            # elif label[x][y] == 2:
            #     rst[x][y] = [0, 0, 255]  # TC (Red)

    label_Gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
    Image.fromarray(label_Gray).save(join(aim_path, case))