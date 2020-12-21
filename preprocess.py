import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    # plt.imshow(decimg[:, :, ::-1])
    # plt.show()
    return decimg[:, :, ::-1]


def gaussian_blur(img, sigma):
    blurred_img = np.zeros(np.shape(img))
    gaussian_filter(img[:, :, 0], output=blurred_img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=blurred_img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=blurred_img[:, :, 2], sigma=sigma)
    # plt.imshow(blurred_img.astype(np.int32))
    # plt.show()
    return blurred_img


if __name__ == '__main__':
    # fake_img = read_img(r'.\images\1_fake\n02381460_1000_fake.png')
    real_img = read_img(r'.\images\0_real\n02381460_1000_real.png')
    # gaussian_blur(fake_img, 2)
    blurred_img = gaussian_blur(real_img, 2)
    # cv2_jpg(fake_img, 50)
    jpged_img = cv2_jpg(real_img, 50)

    plt.subplot(1, 3, 1), plt.imshow(real_img)
    plt.title('original'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(blurred_img.astype(np.int32))
    plt.title('gaussian_blurred'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(jpged_img)
    plt.title('jpged'), plt.axis('off')
    plt.show()
