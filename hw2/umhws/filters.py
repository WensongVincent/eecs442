import os

import numpy as np

from common import read_img, save_img

import pdb
import cv2
import matplotlib.pyplot as plt


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    H, W = image.shape
    h, w = patch_size
    num_h = H // h
    num_w = W // w
    
    for i in range(num_h):
        for j in range(num_w):
            patch = image[i * h : (i + 1) * h, i * w : (i + 1) * w]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            patch = (patch - patch_mean) / patch_std
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            output.append(patch)  
    # import pdb; pdb.set_trace()
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    output = np.zeros_like(image)
    if len(kernel.shape) == 2:
        kernel = kernel[ : :-1 , : :-1]
    elif len(kernel.shape) == 1:
        kernel = kernel[ : :-1]
    
    H, W = image.shape
    h, w = kernel.shape
    
    padding = [h//2, w//2]
    padded_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_image[padding[0] : H + padding[0], padding[1] : W + padding[1]] = image
    
    for y in range(H):
        for x in range(W):
            patch = padded_image[y : y + h , x : x + w] 
            output[y, x] = np.sum(patch * kernel)
    
    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([-1, 0, 1]).reshape(1, 3)  # 1 x 3
    ky = np.array([-1, 0, 1]).reshape(3, 1)  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None
    S_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1],])
    
    S_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1],])
    
    Gx = convolve(image, S_x)
    Gy = convolve(image, S_y)
    grad_magnitude = np.sqrt(Gx ** 2  + Gy ** 2)

    return Gx, Gy, grad_magnitude

def bilateral_filter(image, window_size, sigma_d, sigma_r):
    """
    Return filtered image using a bilateral filter

    Input-  image: H x W
            window_size: (h, w)
            sigma_d: sigma for the spatial kernel
            sigma_r: sigma for the range kernel
    Output- output: filtered image
    """
    # TODO: complete the bilateral filtering, assuming spatial and range kernels are gaussian
    H, W = image.shape
    h, w = window_size
    output = np.zeros_like(image, dtype=image.dtype)
    
    padding = [h//2, w//2]
    padded_image = np.zeros((H + 2 * padding[0], W + 2 * padding[1]), dtype=image.dtype)
    padded_image[padding[0] : H + padding[0], padding[1] : W + padding[1]] = image
    
    range_x = np.arange(-int(w / 2), int(w / 2) + 1)
    range_y = np.arange(-int(h / 2), int(h / 2) + 1)
    mesh_x, mesh_y = np.meshgrid(range_x, range_y)
    dis_mat = mesh_x **2 + mesh_y **2
    # pdb.set_trace()
    
    for y in range(H):
        for x in range(W):
            term1 = - dis_mat / (2 * sigma_d ** 2)
            # pdb.set_trace()
            image_in_kernel = padded_image[y : y + h, x : x + w] 
            term2 = - ( np.linalg.norm((image[y, x] -  image_in_kernel), keepdims=True) ** 2 / (2 * sigma_r ** 2))
            w_ij = np.exp(term1 + term2)
            output[y, x] = (image_in_kernel * w_ij).sum() / w_ij.sum()
            # pdb.set_trace()

    return output


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    idxs = [np.random.randint(0, len(patches)) for _ in range(3)]
    # print(idxs)
    chosen_patches = np.array([patches[i] for i in idxs])
    chosen_patches = chosen_patches.reshape(16, -1)
    # import pdb; pdb.set_trace()
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    kernel_size = 3
    kernel_sigma = 0.572
    # kernel_sigma = 2
    kernel_range = np.arange(-int(kernel_size / 2), int(kernel_size / 2) + 1)
    kernel_x, kernel_y = np.meshgrid(kernel_range, kernel_range)
    kernel_gaussian = np.exp(- (kernel_x ** 2 + kernel_y ** 2) / (2 * kernel_sigma ** 2))
    kernel_gaussian /= kernel_gaussian.sum()
    # print(kernel_gaussian.sum())
    # pdb.set_trace()
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    
    # (h) complete biliateral_filter()
    if not os.path.exists("./bilateral"):
        os.makedirs("./bilateral")

    image_bilataral_filtered = bilateral_filter(img, (5, 5), 3, 75)
    img_cv2 = cv2.imread('./grace_hopper.png')
    image_bilataral_filtered_cv2 = cv2.bilateralFilter(img_cv2, 5, 75, 3)
    save_img(image_bilataral_filtered, "./bilateral/bilateral_output.png")
    save_img(image_bilataral_filtered_cv2, "./bilateral/bilateral_output_cv2.png")

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load('log1d.npz')
    plt.figure(1)
    plt.plot(data['log50'])
    plt.plot(data['gauss53'] - data['gauss50'])
    plt.legend(['Original', 'Approx'])
    plt.show()
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
