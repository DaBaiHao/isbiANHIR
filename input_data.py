#%%
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

DATASET_MEDIUM_DIR = 'dataset_medium.csv'

#%%
def get_files(file_dir):
    imgs_dirs = []
    df = pd.read_csv(DATASET_MEDIUM_DIR)
    print(df.columns)
    for each_img_dir in df['Source image']:
        imgs_dirs.append('images/'+each_img_dir)
    print(imgs_dirs)
    # Image.MAX_IMAGE_PIXELS = 1000000000
    # image = Image.open(imgs_dirs[0])
    # plt.imshow(image)  # 显示图片
    # plt.axis('on')  # 显示坐标轴
    # plt.show()
    Source_landmarks = []
    for each_Source_landmarks in df['Source landmarks']:
        Source_landmarks.append('images/'+each_img_dir)


imgs_dirs = []
df = pd.read_csv(DATASET_MEDIUM_DIR)
for each_img_dir in df['Source image']:
    imgs_dirs.append('images/'+each_img_dir)
Image.MAX_IMAGE_PIXELS = 1000000000
image = Image.open(imgs_dirs[0])
plt.imshow(image) # 显示图片
plt.axis('on') # 显示坐标轴
plt.show()



