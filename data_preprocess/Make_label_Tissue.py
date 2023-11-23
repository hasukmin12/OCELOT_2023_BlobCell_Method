import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import cv2
join = os.path.join
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

train_label_path = "/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/annotations/val/tissue"
label_list = sorted(next(os.walk(train_label_path))[2])

aim_train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/tissue2"
os.makedirs(aim_train_label_path, exist_ok=True)


for case in label_list:
    if case[-4:]==".png":
        print(case)
        bf_label = cv2.imread(join(train_label_path, case))
        # print('{0:03d}.png'.format(img_list.index(case)+1))
        # bf_label = cv2.cvtColor(cv2.imread(join(train_label_path, '{0:03d}.png'.format(img_list.index(case)+401))), cv2.COLOR_BGR2RGB) # 538
        
        # print('cell_{0:03d}.png'.format(int(case[:3])))
        label = np.zeros((1024, 1024), dtype=np.uint8)
        print(bf_label.max())
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                if bf_label[x][y][0] == 1: # BGR
                    label[x][y] = 1  # BGR  - Background
                elif bf_label[x][y][0] == 2: # BGR
                    label[x][y] = 2  # BGR  - Cancer Area
                elif bf_label[x][y][0] == 255: # BGR
                    label[x][y] = 3  # BGR  - Unknown
        
        Image.fromarray(label).save(join(aim_train_label_path, 'cell_{0:03d}.png'.format(int(case[:3])-1)))

