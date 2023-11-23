#%%
from PIL import Image
from platform import python_version
import cv2
import numpy as np
import tifffile as tif


# test a single image
img = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/cell/cell_537.png'
# gt = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/BlobCell/cell_400.png'
# rst = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/tissue_crop2/cell_400.png'
gt = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/BlobCell_one_label/cell_537.png'
rst = '/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/test/tissue_crop2/cell_537.png'






# Let's take a look at the dataset
import matplotlib.pyplot as plt

img = np.array(Image.open(img))
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.show()

print(img.shape)


gt = np.array(Image.open(gt))
plt.figure(figsize=(8, 6))
plt.imshow(gt)
plt.show()

print(gt.shape)
print(gt.max())
print(gt.ndim)
print(gt.dtype)



rst = np.array(Image.open(rst))
plt.figure(figsize=(8, 6))
plt.imshow(rst)
plt.show()

print(rst.shape)
print(rst.max())
print(rst.ndim)
print(rst.dtype)



# rst_png = mmcv.imread(rst_png)
# rst_png = rst_png*100
# plt.figure(figsize=(8, 6))
# plt.imshow(mmcv.bgr2rgb(rst_png))
# plt.show()

# print(rst_png.shape)
# print(rst_png.max())
# print(rst_png.ndim)
# print(rst_png.dtype)
# %%
