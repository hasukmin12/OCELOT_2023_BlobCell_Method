#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmcv

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed, random_walker
from skimage.feature import peak_local_max


# follow belowed link
# https://github.com/Connor323/Cancer-Cell-Tracking/blob/master/Code/watershed.py




def TC_value(image):

    orgin = image
    b = image.shape[0]
    total_tc = []

    for b1 in range(b):

        image = orgin[b1,0].detach().cpu().numpy()

        image2 = np.zeros(image.shape, dtype=np.uint8)

        # blob 분리하는 코드
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if image[x][y] == 1:
                    image[x][y] = 255
                elif image[x][y] == 2:
                    image[x][y] = 0
                    image2[x][y] = 255  

        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 0)
        # result_dist_transform = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _, sure_fg = cv2.threshold(dist_transform, 4, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(image, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(image, markers)


        


        dist_transform2 = cv2.distanceTransform(image2, cv2.DIST_L2, 0)
        # result_dist_transform2 = cv2.normalize(dist_transform2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _, sure_fg2 = cv2.threshold(dist_transform2, 0.1*dist_transform2.max(), 255, cv2.THRESH_BINARY)
        sure_fg2 = np.uint8(sure_fg2)
        unknown2 = cv2.subtract(image2, sure_fg2)

        # Marker labelling
        _, markers2 = cv2.connectedComponents(sure_fg2)
        # Add one to all labels so that sure background is not 0, but 1
        markers2 = markers2+1
        # Now, mark the region of unknown with zero
        markers2[unknown2==255] = 0

        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        markers2 = cv2.watershed(image2, markers2)


        c_1 = markers.max()
        c_2 = markers2.max()
        # print("Tumor : ", c_1)
        # print("Non-Tumor : ", c_2)

        tc = (c_1 / (c_1 + c_2)) * 100
        total_tc.append(tc)

    # normalize list
    min_val = min(total_tc)
    max_val = max(total_tc)
    range_val = max_val - min_val
    normalize_total_tc = [(x - min_val) / range_val for x in total_tc]

    return total_tc








def TC_value_for_logit_map(image):

    image = image.detach().cpu().numpy()
    orgin = image
    b = image.shape[0]
    total_tc = []

    for b1 in range(b):

        image = orgin[b1]
        image = np.argmax(image, axis=0).astype(np.uint8)
        image2 = np.zeros(image.shape, dtype=np.uint8)

        # blob 분리하는 코드
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if image[x][y] == 1:
                    image[x][y] = 255
                elif image[x][y] == 2:
                    image[x][y] = 0
                    image2[x][y] = 255 

        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 0)
        # result_dist_transform = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _, sure_fg = cv2.threshold(dist_transform, 4, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(image, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(image, markers)


        


        dist_transform2 = cv2.distanceTransform(image2, cv2.DIST_L2, 0)
        # result_dist_transform2 = cv2.normalize(dist_transform2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _, sure_fg2 = cv2.threshold(dist_transform2, 0.1*dist_transform2.max(), 255, cv2.THRESH_BINARY)
        sure_fg2 = np.uint8(sure_fg2)
        unknown2 = cv2.subtract(image2, sure_fg2)

        # Marker labelling
        _, markers2 = cv2.connectedComponents(sure_fg2)
        # Add one to all labels so that sure background is not 0, but 1
        markers2 = markers2+1
        # Now, mark the region of unknown with zero
        markers2[unknown2==255] = 0

        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        markers2 = cv2.watershed(image2, markers2)


        c_1 = markers.max()
        c_2 = markers2.max()
        # print("Tumor : ", c_1)
        # print("Non-Tumor : ", c_2)

        tc = (c_1 / (c_1 + c_2)) * 100
        total_tc.append(tc)

    # normalize list
    # min_val = min(total_tc)
    # max_val = max(total_tc)
    # range_val = max_val - min_val
    # normalize_total_tc = [(x - min_val) / range_val for x in total_tc]


    return total_tc
# %%



def calculate_TC_loss(y, x):
    loss_c = []
    for a in range(len(x)):
        loss = abs((y[a] - x[a]) / y[a])
        loss_c.append(loss)
    
    return np.mean(loss_c)