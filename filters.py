import numpy as np
import os
from common import *


def convolve(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    kernel = np.flipud(np.fliplr(kernel))
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros(image.shape)
    
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            
            output[i, j] = np.sum(region * kernel)
    
    return output


def edge_detection(image):
    kx = np.array([[0.5, 0, -0.5]]) 
    ky = np.array([[0.5], [0], [-0.5]])  

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return grad_magnitude, Ix, Iy


def sobel_operator(image):
    kx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    ky = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    Gx = convolve(image, kx)
    Gy = convolve(image, ky)
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    output = []
    
    for alpha in angles:
        kernel = np.array([[np.cos(alpha)+np.sin(alpha), 2*np.sin(alpha), -np.cos(alpha)+np.sin(alpha)],
                           [2*np.cos(alpha), 0, -2*np.cos(alpha)],
                           [np.cos(alpha)-np.sin(alpha), -2*np.sin(alpha), -np.cos(alpha)-np.sin(alpha)]])
        output.append(convolve(image, kernel))

    return output


def main():
    img = read_img('./data/grace_hopper.png')

    kernel_gaussian = np.zeros((3,3))
    var = 1/(2*np.log(2))
    for i in range(3):
        for j in range(3):
            kernel_gaussian[i,j] = 1/(2*np.pi*var)*np.exp(-((i-1)**2+(j-1)**2)/(2*var))

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./results/gaussian_filtered.png")

    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./results/edge_detection_original.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./results/edge_detection_gaussian_filtered.png")

    print("Gaussian Filter is done. ")

    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./results/sobel_operator_Gx.png")
    save_img(Gy, "./results/sobel_operator_Gy.png")
    save_img(edge_sobel, "./results/edge_detection_sobel.png")

    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./results/sobel_operator_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")

    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./results/log_filter_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./results/log_filter_LoG2.png")

    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()