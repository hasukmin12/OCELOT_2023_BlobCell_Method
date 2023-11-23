import os
import numpy as np
# import mmcv
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
import json
from pathlib import Path
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List
from skimage import exposure
import tifffile as tif
join = os.path.join

def read_json(fpath: Path) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data 
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data



img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/tissue"
img_list = sorted(next(os.walk(img_path))[2])

aim_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/tissue_crop"
os.makedirs(aim_path, exist_ok=True)

meta_path = "/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/metadata.json"
meta = read_json(meta_path)

if "Thumbs.db" in img_list:
    img_list.remove("Thumbs.db")

for case in img_list:
    if case[-4:]==".png":
        print(case)
        if case != 'cell_585.png':
            if case != 'cell_588.png':
                if case != 'cell_608.png':
                    if case != 'cell_614.png':
        # print("{0:03d}".format(int(img_list.index(case)+1)))
        # print(meta['sample_pairs']["{0:03d}".format(int(img_list.index(case)+1))])
        # cx = meta['sample_pairs']["{0:03d}".format(int(img_list.index(case)+81))]['patch_x_offset']
        # cy = meta['sample_pairs']["{0:03d}".format(int(img_list.index(case)+81))]['patch_y_offset']
                        cx = meta['sample_pairs']["{0:03d}".format(int(img_list.index(case)+538))]['patch_x_offset'] # +1 , +401, +538 
                        cy = meta['sample_pairs']["{0:03d}".format(int(img_list.index(case)+538))]['patch_y_offset']

                        # 256*256 박스 생성 중심점(cx, cy)
                        tissue_patch = np.array(Image.open(join(img_path, case)))
                        x_i = int(1024*cx-128)
                        x_o = int(1024*cx+128)
                        y_i = int(1024*cy-128)
                        y_o = int(1024*cy+128)

                        crop_patch = tissue_patch[y_i:y_o, x_i:x_o]

                        # 256*256 -> 1024*1024
                        rst = cv2.resize(crop_patch, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

                        Image.fromarray(rst).save(join(aim_path, case))


        





        

    

