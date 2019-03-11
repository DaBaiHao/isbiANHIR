import cv2
import numpy as np

print(cv2.__version__)
help(cv2.xfeatures2d)
sift = cv2.xfeatures2d.SIFT_create()
