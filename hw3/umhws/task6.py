"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    X = desc1
    Y = desc2

    X_norm_sq = np.linalg.norm(X, axis=1, keepdims=True) ** 2
    Y_norm_sq = np.linalg.norm(Y, axis=1, keepdims=True) ** 2
    dist = np.sqrt(np.maximum(0, (X_norm_sq + Y_norm_sq.T - 2 * (X @ Y.T))))
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    matches = []
    
    dist = compute_distance(desc1, desc2)
    idx_smallest_two = np.argsort(dist, axis=1)[:, :2]
    ratio = np.take_along_axis(dist, idx_smallest_two, axis=1)[:, 0] / np.take_along_axis(dist, idx_smallest_two, axis=1)[:, 1]
    
    idx_ii = np.where((ratio < ratioThreshold))[0]
    idx_jj = idx_smallest_two[idx_ii, 0]
    matches = np.hstack((idx_ii[:, np.newaxis], idx_jj[:, np.newaxis]))
    # import pdb; pdb.set_trace()
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints
    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations
    output = np.vstack((img1, img2))
    H1, W1, _ = img1.shape
    kps = common.get_match_points(kp1, kp2, matches)
    for i in range(kps.shape[0]):
        p1 = kps[i, :2].astype(int)
        p2 = (kps[i, 2:] + np.array([0, H1])).astype(int)
        # print(p1, p2)
        cv2.line(output, (p1), (p2), (0, 0, 255), 4 )
    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''
    # Get dimensions of input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Corners of img1
    corners_img1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Corners of img2 transformed by H
    corners_img2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners_img2_transformed = cv2.perspectiveTransform(corners_img2, H)
    
    # Combine the corners
    all_corners = np.concatenate((corners_img1, corners_img2_transformed), axis=0)
    
    # Find the bounding rectangle
    x_min, y_min = np.intp(np.min(all_corners, axis=0).ravel() - 0.5)
    x_max, y_max = np.intp(np.max(all_corners, axis=0).ravel() + 0.5)
    
    # Translation homography
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]], dtype=np.float32)

    # Warp both images
    warp_img1 = cv2.warpPerspective(img1, H_translation, (x_max - x_min, y_max - y_min))
    warp_img2 = cv2.warpPerspective(img2, H_translation.dot(H.astype(np.float32)), (x_max - x_min, y_max - y_min))

    # Create a mask of the combined size for where img1 and warped img2 are not zero
    mask_img1 = np.sum(warp_img1, axis=2) > 0
    mask_img2 = np.sum(warp_img2, axis=2) > 0
    mask_overlap = mask_img1 & mask_img2
    mask_img1_only = mask_img1 & ~mask_overlap
    mask_img2_only = mask_img2 & ~mask_overlap
    
    # Initialize the stitched image canvas
    stitched_img = np.zeros_like(warp_img1)
    
    # Place each image on the canvas according to the masks
    stitched_img[mask_img1_only] = warp_img1[mask_img1_only]
    stitched_img[mask_img2_only] = warp_img2[mask_img2_only]
    
    # Handle overlapping areas
    stitched_img[mask_overlap] = warp_img1[mask_overlap] // 2 + warp_img2[mask_overlap] // 2
    
    return stitched_img


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image
    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    
    kp1, desc1 = common.get_AKAZE(I1)
    kp2, desc2 = common.get_AKAZE(I2)
    
    ratio = 0.7
    matches = find_matches(desc1, desc2, ratio)
    kps = common.get_match_points(kp1, kp2, matches)
    
    H = RANSAC_fit_homography(kps, eps= 4, nIters=2000)
    print(H)
    
    stitched = warp_and_combine(img2, img1, H)
    
    return stitched 


if __name__ == "__main__":

    #Possible starter code; you might want to loop over the task 6 images
    # to_stitch = 'lowetag'
    to_stitch = 'eynsham'
    I1 = read_img(os.path.join('task6',to_stitch,'p1.jpg'))
    I2 = read_img(os.path.join('task6',to_stitch,'p2.jpg'))
    
    # kp1, desc1 = common.get_AKAZE(I1)
    # kp2, desc2 = common.get_AKAZE(I2)
    
    # ratio = 0.7
    # matches = find_matches(desc1, desc2, ratio)
    
    # draw = draw_matches(I1, I2, kp1, kp2, matches)
    # save_img(draw, "draw_matches.jpg")
    
    res = make_warped(I1,I2)
    save_img(res,"result_"+to_stitch+".jpg")
