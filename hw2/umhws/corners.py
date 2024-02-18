import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    output = np.zeros_like(image)
    H, W = image.shape
    h, w = window_size    
    
    shifted_image = np.roll(image, (u, v), axis=(1, 0))
    
    padding = (h//2, w//2)
    
    padded_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_image[padding[0] : padding[0] + H, padding[1] : padding[1] + W] = image
    padded_shifted_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_shifted_image[padding[0] : padding[0] + H, padding[1] : padding[1] + W] = shifted_image
    
    for y in range(H):
        for x in range(W):
            e = np.sum((padded_shifted_image[y : y + h, x : x + w] - padded_image[y : y + h , x : x + h]) ** 2)
            output[y, x] = e
    
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    kx = np.array([-1, 0, 1]).reshape(1, 3)
    ky = np.array([-1, 0, 1]).reshape(3, 1)
    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    M = np.zeros((3, image.shape[0], image.shape[1]))
    
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         # import pdb; pdb.set_trace()
    kernel = np.ones(window_size)
    M[0] = scipy.ndimage.convolve(Ixx, kernel, mode='constant', cval=0)
    M[1] = scipy.ndimage.convolve(Ixy, kernel, mode='constant', cval=0)
    M[2] = scipy.ndimage.convolve(Iyy, kernel, mode='constant', cval=0)
    
    alpha = 0.05
    
    response = M[0] * M[2] - M[1] ** 2 - alpha * (M[0] + [2]) ** 2

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    W = (5, 5)
    tuples = ((0, 5), (0, -5), (5, 0), (-5, 0))
    for i, (u, v) in enumerate(tuples):
        score = corner_score(img, u, v, W)
        save_img(score, f"./feature_detection/corner_score_{i}.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
