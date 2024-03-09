"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import homography_transform, RANSAC_fit_homography
import cv2
import os

from task6 import *

def task7_warp_and_combine(img1, img2, H):
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
                but make sure in V, for pixels covered by both img1 and warped img2,
                you see only img2
    '''
    # Warp img2 onto img1's plane
    warp_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    # Create mask of where the warped image is non-zero
    mask = (warp_img2.sum(-1) > 0)
    # Initialize output image
    V = img1.copy()
    # Place img2 on the masked regions of img1
    V[mask] = warp_img2[mask]
    
    return V

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    # augment = None
    # Resize transfer image to the template's size
    transfer = cv2.resize(transfer, (template.shape[1], template.shape[0]))
    
    kp1, desc1 = common.get_AKAZE(template)
    kp2, desc2 = common.get_AKAZE(scene)
    
    ratio = 0.7
    matches = find_matches(desc1, desc2, ratio)
    kps = common.get_match_points(kp1, kp2, matches)
    
    H = RANSAC_fit_homography(kps, eps= 4, nIters=2000)
    
    augment = task7_warp_and_combine(scene, transfer, H)
    
    return augment

if __name__ == "__main__":
    # Task 7
    scene_img_path = 'task7/scenes/lacroix/scene.jpg' 
    template_img_path = 'task7/scenes/lacroix/template.png'
    transfer_img_path = 'task7/seals/monk.png'
    # scene_img_path = 'task7/scenes/bbb/scene.jpg' 
    # template_img_path = 'task7/scenes/bbb/template.png'
    # transfer_img_path = 'task7/seals/um.png'

    scene = read_img(scene_img_path)
    template = read_img(template_img_path)
    transfer = read_img(transfer_img_path)

    improved_image = improve_image(scene, template, transfer)

    save_img(improved_image, f'improved_lacroix.jpg' )
