#%%
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

DATASET_MEDIUM_DIR = 'dataset_medium.csv'
# expected image extensions
IMAGE_EXT = ('.png', '.jpg', '.jpeg')
LANDMARK_COORDS = ('X', 'Y')
Image.MAX_IMAGE_PIXELS = None
#%%
# def get_files(file_dir):
#     imgs_dirs = []
#     df = pd.read_csv(DATASET_MEDIUM_DIR)
#     print(df.columns)
#     for each_img_dir in df['Source image']:
#         imgs_dirs.append('images/'+each_img_dir)
#     print(imgs_dirs)
#     # Image.MAX_IMAGE_PIXELS = 1000000000
#     # image = Image.open(imgs_dirs[0])
#     # plt.imshow(image)  # 显示图片
#     # plt.axis('on')  # 显示坐标轴
#     # plt.show()
#     Source_landmarks = []
#     for each_Source_landmarks in df['Source landmarks']:
#         Source_landmarks.append('images/'+each_img_dir)


imgs_dirs = []
dataset_read_result = pd.read_csv(DATASET_MEDIUM_DIR)
i = 0
for each_img_dir,\
    each_landmarks_dir,\
    each_target_image,\
    each_target_landmarks,\
    each_status in zip(dataset_read_result['Source image'],
                       dataset_read_result['Source landmarks'],
                       dataset_read_result['Target image'],
                       dataset_read_result['Target landmarks'],
                       dataset_read_result['status']):
    each_img_dir = 'images/'+each_img_dir
    each_landmarks_dir = 'landmarks/' + each_landmarks_dir
    each_target_image = 'images/'+each_target_image
    each_target_landmarks = 'landmarks/' + each_target_landmarks

    dataset_read_result.set_value(index=i, col='Source image', value=each_img_dir)
    dataset_read_result.set_value(index=i, col='Source landmarks', value=each_landmarks_dir)
    dataset_read_result.set_value(index=i, col='Target image', value=each_target_image)
    dataset_read_result.set_value(index=i, col='Target landmarks', value=each_target_landmarks)

    imgs_dirs.append(each_img_dir)
    i = i+1

print(dataset_read_result['Source image'][1])
print(imgs_dirs[1])
# image = Image.open(imgs_dirs[0])
# plt.imshow(image) # 显示图片
# plt.axis('on') # 显示坐标轴
# plt.show()

#%%
import os
source_image_array = dataset_read_result['Source image']
target_image_array = dataset_read_result['Target image']
assert len(source_image_array) == len(target_image_array)






