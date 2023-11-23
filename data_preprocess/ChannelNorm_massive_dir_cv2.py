import argparse
import numpy as np
from PIL import Image
import os
from skimage import io, segmentation, morphology, measure, exposure
join = os.path.join
import cv2

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageDir', type=str, default='/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/images/train/cell', help='RGB image file')
    parser.add_argument('--saveDir', type=str, default='/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/ChannelNorm_cv2/train/cell', help='save file')
    parser.add_argument('--Io', type=int, default=240)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.15)
    args = parser.parse_args()
    
    input_list = sorted(next(os.walk(args.imageDir))[2])
    os.makedirs(args.saveDir, exist_ok=True)
    if "Thumbs.db" in input_list:
        input_list.remove("Thumbs.db")

    for case in input_list:
        print(case)
        img = np.array(Image.open(join(args.imageDir, case)))
        output_path = join(args.saveDir, 'cell_{0:03d}.png'.format(input_list.index(case))) # +400
        
        pre_img_data = np.zeros(img.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = img[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        
        # Image.fromarray(pre_img_data).save(output_path)
        cv2.imwrite(output_path, pre_img_data)
        
