import cv2
import glob
import numpy as np
import os


IR_images = sorted(glob.glob('triandata/IR/*.jpg'))
RGB_images = sorted(glob.glob('triandata/Flir_RGB/*.jpg'))

for i in np.arange(len(RGB_images)):
    IR_img = cv2.imread(IR_images[i])
    RGB_img = cv2.imread(RGB_images[i])

    sss = os.path.join("D:/PycharmProjects/IR_RGB registration/triandata/Flir_Merged", RGB_images[i][19::])

    dst = cv2.addWeighted(IR_img, 0.5, RGB_img, 0.5, 0)
    # cv.imshow('warp_RGB_img', RGB_img_warp)
    # cv.imshow(IR_images[i], IR_img)
    cv2.imshow('fused_img', dst)
    cv2.imwrite(sss, dst)
    # key = cv2.waitKey(0)
    # if key == 'q':
    #     exit()
    cv2.waitKey(50)
    cv2.destroyAllWindows()