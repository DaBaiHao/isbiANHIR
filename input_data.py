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


def warp_affine(img1, img2, lnd1, lnd2):
    """ estimate an affine transform and perform image and landmarks warping
    :param ndarray img1: reference image
    :param ndarray img2: moving landmarks
    :param ndarray lnd1: reference image
    :param ndarray lnd2: moving landmarks
    :return (ndarray, ndarray): moving image and landmarks warped to reference
    """
    nb = min(len(lnd1), len(lnd2))
    pts1 = lnd1[list(LANDMARK_COORDS)].values[:nb]
    pts2 = lnd2[list(LANDMARK_COORDS)].values[:nb]
    _, matrix_inv, _, pts2_warp = estimate_affine_transform(pts1, pts2)
    lnd2_warp = pd.DataFrame(pts2_warp, columns=LANDMARK_COORDS)
    matrix_inv = matrix_inv[:2, :3].astype(np.float64)
    img2_warp = cv.warpAffine(img2, matrix_inv, img1.shape[:2][::-1])
    return img2_warp, lnd2_warp
