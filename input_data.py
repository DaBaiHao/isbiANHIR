#%%
import  pandas as pd
import numpy as np

DATASET_MEDIUM_DIR = 'dataset_medium.csv'

#%%
def get_files(file_dir):
    imgs_dirs = []
    df = pd.read_csv(DATASET_MEDIUM_DIR)
    print(df.columns)
    for each_img_dir in df['Source image']:
        imgs_dirs.append('images/'+each_img_dir)
    print(imgs_dirs)


imgs_dirs = []
df = pd.read_csv(DATASET_MEDIUM_DIR)
for each_img_dir in df['Source image']:
    imgs_dirs.append('images/'+each_img_dir)
# print(imgs_dirs)
