import numpy as np
import cv2 as cv
import glob
import os


# 检查一个旋转矩阵是否有效
def isRotationMatrix(R) :
    # 得到该矩阵的转置
    Rt = np.transpose(R)
    # 旋转矩阵的一个性质是，相乘后为单位阵
    shouldBeIdentity = np.dot(Rt, R)
    # 构建一个三维单位阵
    I = np.identity(3, dtype = R.dtype)
    # 将单位阵和旋转矩阵相乘后的值做差
    n = np.linalg.norm(I - shouldBeIdentity)
    # 如果小于一个极小值，则表示该矩阵为旋转矩阵
    return n < 1e-6


# 这部分的代码输出与Matlab里边的rotm2euler一致
def rotationMatrixToEulerAngles(R):
    # 断言判断是否为有效的旋转矩阵
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([z, y, x])/np.pi*180


H_matrx = np.zeros([1, 3, 3])

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Arrays to store object points and image points from all the images.
IR_points = [] # 3d point in real world space
RGB_points = [] # 2d points in image plane.
IR_images = sorted(glob.glob('data/IR_2/*.png'))
RGB_images = sorted(glob.glob('data/RGB_2/*.jpg'))
K = np.array([[1.73550154e+03, 0.00000000e+00, 6.58548462e+02],
              [0.00000000e+00, 1.73278080e+03, 5.20006510e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
for i in np.arange(len(RGB_images)):
    IR_img = cv.imread(IR_images[i])
    RGB_img = cv.imread(RGB_images[i])

    IR_gray = cv.cvtColor(IR_img, cv.COLOR_BGR2GRAY)
    RGB_gray = cv.cvtColor(RGB_img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    IR_ret, IR_corners = cv.findChessboardCorners(IR_gray, (11, 8), corners=88, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    RGB_ret, RGB_corners = cv.findChessboardCorners(RGB_gray, (11, 8), corners=88, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    # If found, add object points, image points (after refining them)
    if IR_ret == True and RGB_ret == True:

        IR_corners2 = cv.cornerSubPix(IR_gray, IR_corners, (3, 3), (-1, -1), criteria)
        RGB_corners2 = cv.cornerSubPix(RGB_gray, RGB_corners, (7, 7), (-1, -1), criteria)

        IR_points.append(IR_corners2)
        IR_pts = np.array(IR_points)[0]

        RGB_points.append(RGB_corners2)
        RGB_pts = np.array(RGB_points)[0]

        M, mask = cv.findHomography(RGB_corners2, IR_corners2, method=0)
        matchesMask = mask.ravel().tolist()

        if np.sum(np.array(matchesMask)) >= 88:

            H_matrx = np.concatenate((H_matrx, np.expand_dims(M, axis=0)), axis=0)

            print(IR_images[i], RGB_images[i])
            # print(M)
            # Draw and display the corners
            cv.drawChessboardCorners(IR_img, (11, 8), IR_corners2, IR_ret)
            cv.drawChessboardCorners(RGB_img, (11, 8), RGB_corners2, RGB_ret)

            # img_for_show = np.hstack([RGB_img, IR_img])
            cv.imshow(RGB_images[i], RGB_img)
            cv.imshow(IR_images[i], IR_img)

            RGB_img_warp = cv.warpPerspective(RGB_img, M, (IR_img.shape[1], IR_img.shape[0]))
            cv.imshow('warp_RGB_img', RGB_img_warp)

            key = cv.waitKey(0)
            if (key == 'q'):
                exit()
            cv.destroyAllWindows()

            # img_draw_matches = cv.hconcat([IR_img, RGB_img])
            # for i in range(len(IR_corners2)):
            #     pt1 = np.array([IR_corners2[i][0], IR_corners2[i][1], 1])
            #     pt1 = pt1.reshape(3, 1)
            #     pt2 = np.dot(M, pt1)
            #     pt2 = pt2 / pt2[2]
            #     end = (int(IR_img.shape[1] + pt2[0]), int(pt2[1]))
            #     cv.line(img_draw_matches, tuple([int(j) for j in IR_corners2[i]]), end, cv.randomColor(), 2)
            # cv.imshow("Draw matches", img_draw_matches)
            # cv.waitKey(0)

            num, Rs, Ts, Ns = cv.decomposeHomographyMat(M, K)

            # print(Rs[0])
            # print(cv.Rodrigues(Rs[0])[0])
            # print(rotationMatrixToEulerAngles(Rs[0]))
            print(Ts[0])


# cv.destroyAllWindows()
# print(H_matrx)
np.save("H_matrx.npy", H_matrx)
