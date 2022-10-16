import cv2
import numpy as np
import cv2 as cv
import glob
import timeit
from numpy.fft import fft2, ifft2, fftshift


# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(input_im1, input_im2, mode=cv2.MOTION_TRANSLATION, num_of_iters=500, term_eps=1e-4):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(input_im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(input_im2, cv2.COLOR_BGR2GRAY)

    cv.imshow("Image 1", im1_gray)
    cv.imshow("Image 2", im2_gray)

    key = cv.waitKey(0)
    if (key == 'q'):
        exit()
    cv.destroyAllWindows()
    # Find size of image1
    sz = input_im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_of_iters, term_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, mode, criteria)

    if mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(input_im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(input_im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    cv.imshow("Image 1", input_im1)
    cv.imshow("Image 2", input_im2)
    cv.imshow("Aligned Image 2", im2_aligned)
    # cv.waitKey(0)
    #
    # dst = cv.addWeighted(RGB_img_warp, 0.5, IR_img, 0.5, 0)
    # # cv.imshow('warp_RGB_img', RGB_img_warp)
    # # cv.imshow(IR_images[i], IR_img)
    # cv.imshow('fused_img', dst)
    key = cv.waitKey(0)
    if key == 'q':
        exit()
    cv.destroyAllWindows()

    return im2_aligned, warp_matrix


# FFT phase correlation
def translation(input_im1, input_im2):
    # Convert images to grayscale
    im0 = cv2.cvtColor(input_im1, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(input_im2, cv2.COLOR_BGR2GRAY)

    cv.imshow("Image 1", im0)
    cv.imshow("Image 2", im1)

    key = cv.waitKey(0)
    if (key == 'q'):
        exit()
    cv.destroyAllWindows()

    rows, cols = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), [rows, cols])
    if t0 > rows // 2:
        t0 -= rows
    if t1 > cols // 2:
        t1 -= cols

    m_tran = np.float32([[1, 0, t1], [0, 1, t0]])
    im2_aligned = cv2.warpAffine(input_im2, m_tran, (cols, rows))

    # Show final results
    cv.imshow("Image 1", input_im1)
    cv.imshow("Image 2", input_im2)
    cv.imshow("Aligned Image 1", im2_aligned)
    # cv.waitKey(0)
    #
    dst = cv.addWeighted(input_im1, 0.5, im2_aligned, 0.5, 0)
    # cv.imshow('warp_RGB_img', RGB_img_warp)
    # cv.imshow(IR_images[i], IR_img)
    cv.imshow('fused_img', dst)
    key = cv.waitKey(0)
    if key == 'q':
        exit()
    cv.destroyAllWindows()

    return [t0, t1]


IR_images = sorted(glob.glob('data/IR_2/*.png'))
RGB_images = sorted(glob.glob('data/RGB_2/*.jpg'))

H_matrix_set = np.load("H_matrx.npy")
# aaa = M[1, :, :]

for i in np.arange(len(RGB_images)):
    IR_img = cv.imread(IR_images[i])
    RGB_img = cv.imread(RGB_images[i])

    RGB_img_warp = cv.warpPerspective(RGB_img, H_matrix_set[5, :, :], (IR_img.shape[1], IR_img.shape[0]))

    print(RGB_images[i][11::])
    cv2.imwrite("./data/wraped_RGB_2/" + RGB_images[i][11::], RGB_img_warp)

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    initial_matrix = H_matrix_set[5, :, :]
    initial_matrix = initial_matrix.astype(np.float32)

    # a, b = translation(IR_img, RGB_img_warp)
    # print(a, b)

