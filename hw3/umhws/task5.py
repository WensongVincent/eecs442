"""
Task 5 Code
"""
import numpy as np
from matplotlib import pyplot as plt
from common import save_img, read_img
from homography import fit_homography, homography_transform
import os
import cv2


def make_synthetic_view(img, corners, size):
    '''
    Creates an image with a synthetic view of selected region in the image
    from the front. The region is bounded by a quadrilateral denoted by the
    corners array. The size array defines the size of the final image.

    Input - img: image file of shape (H,W,3)
            corner: array containing corners of the book cover in 
            the order [top-left, top-right, bottom-right, bottom-left]  (4,2)
            size: array containing size of book cover in inches [height, width] (1,2)

    Output - A fronto-parallel view of selected pixels (the book as if the cover is
            parallel to the image plane), using 100 pixels per inch.
    '''
    # The desired coordinates for the book corners
    h, w = size
    # Convert from inches to pixels: 1 inch is 100 pixels
    h, w = h * 100, w * 100
    dst_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
    XY = np.hstack((corners, dst_points))
    
    # Compute the homography matrix
    h_matrix = fit_homography(XY)

    # Perform the warp perspective
    warped_image = cv2.warpPerspective(img, h_matrix, (int(w), int(h)))
    return warped_image
    
if __name__ == "__main__":
    # Task 5

    case_name = "threebody"

    I = read_img(os.path.join("task5",case_name,"book.jpg"))
    corners = np.load(os.path.join("task5",case_name,"corners.npy"))
    size = np.load(os.path.join("task5",case_name,"size.npy"))
#     import pdb; pdb.set_trace()

    result = make_synthetic_view(I, corners, tuple(size[0]))
    save_img(result, case_name+"_frontoparallel.jpg")

    
    case_name = "palmer"

    I = read_img(os.path.join("task5",case_name,"book.jpg"))
    corners = np.load(os.path.join("task5",case_name,"corners.npy"))
    size = np.load(os.path.join("task5",case_name,"size.npy"))
#     import pdb; pdb.set_trace()

    result = make_synthetic_view(I, corners, tuple(size[0]))
    save_img(result, case_name+"_frontoparallel.jpg")
