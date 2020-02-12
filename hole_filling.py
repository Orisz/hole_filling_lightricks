import argparse
from PIL import Image
import numpy as np
from weight_func import *  # replace this imnport to use different weight func
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def parse_args_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', help='power of the norm', type=int, default=3)
    parser.add_argument('--input_img', help='input image path', default='Input/lena.jpg')
    parser.add_argument('--input_mask', help='mask path', default='Input/lena_mask_5.png')
    parser.add_argument('--eps', help='epsilon', type=float, default=1e-3)
    parser.add_argument('--connectivity', help='type 4 or 8', type=int, default=4)
    parser.add_argument('--out_img', default='recon_image.png', help='folder to output image')
    args = parser.parse_args()
    return args


def read_images(im_path, mask_path):
    # Read image
    img = Image.open(im_path)
    img_mask = Image.open(mask_path)
    assert img.size == img_mask.size, "input image and mask image size must match!"
    return img, img_mask


def rgb_2_gray_np(im, im_mask):
    # Read image
    img_gray = im.convert('L')
    img_mask_gray = im_mask.convert('L')
    im_np = np.array(img_gray)
    im_mask_np = np.array(img_mask_gray)
    return im_np, im_mask_np


def convert_mask_2_binary(mask):
    assert type(mask).__module__ == np.__name__
    thresh = (np.amax(mask) + np.amin(mask)) / 2.0
    idx = mask > thresh
    mask.fill(1)
    mask[idx] = 0
    return mask


def normalize_image(im):
    im = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
    return im


def preprocess():
    # z, im_path, mask_path, eps, out_path = parse_args_()
    args = parse_args_()
    z = args.z
    input_img_path = args.input_img
    connectivity = args.connectivity
    assert connectivity == 4 or connectivity == 8, "only 4 or 8 pixel connectivity is supported!"
    input_mask = args.input_mask
    eps = args.eps
    out_path = args.out_img
    # read image and mask image
    im, mask = read_images(input_img_path, input_mask)
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    plt.title('original')
    plt.subplot(222)
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])
    plt.title('mask')

    im_np, mask_np = rgb_2_gray_np(im, mask)
    binary_mask = convert_mask_2_binary(mask_np)
    im_np = normalize_image(im_np)
    hole_im_plot = im_np * binary_mask #for plots
    save_and_plot_image(hole_im_plot, 'masked_image.png', 223, plt)
    hole_im = im_np
    hole_im[binary_mask == 0] = -1
    return hole_im, input_img_path, eps, z, connectivity, out_path, plt


def save_and_plot_image(hole_im, name, sub_plot_place, plt):
    img = (hole_im * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(name)
    plt.subplot(sub_plot_place)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    filename, _ = os.path.splitext(name)
    plt.title(filename)


def get_boundary_idx(hole_im, connectivity):
    boundary_idx_y = []
    boundary_idx_x = []
    if connectivity == 4:
        neighbours_idx = lambda y, x: (np.array([y-1, y+1, y, y]), np.array([x, x, x-1, x+1]))
    # connectivity == 8
    else:
        neighbours_idx = lambda y, x:  (np.array([y-1, y+1, y, y, y+1, y+1, y-1, y-1]),
                                        np.array([x, x, x-1, x+1, x+1, x-1, x+1, x-1]))
    for y in range(1, hole_im.shape[0]-1):
        for x in range(1, hole_im.shape[1]-1):
            if hole_im[y, x] == -1:
                # it is hole pixel
                continue
            n_idx = neighbours_idx(y, x)
            neighbours = hole_im[n_idx]
            if -1 in neighbours:
                boundary_idx_y.append(y)
                boundary_idx_x.append(x)
    boundary_idx = (np.array(boundary_idx_y), np.array(boundary_idx_x))
    return boundary_idx


def get_holes_idx(hole_im):
    idx = np.argwhere(hole_im == -1)
    return idx


def fill_hole(hole_im, boundary_idx, holes_idx, weight_class):
    # deep copy
    recon_im = np.copy(hole_im)
    I_v = hole_im[boundary_idx]
    for cur_hole_idx in tqdm(holes_idx):
        weights = [weight_class.get_weight(cur_hole_idx, (y, x)) for y, x in zip(boundary_idx[0], boundary_idx[1])]
        nom = sum(weights * I_v)
        demon = sum(weights)
        val = nom / demon
        recon_im[cur_hole_idx[0], cur_hole_idx[1]] = val
        # if i%100 == 0:
        #     print(f'done with hole pix: {i}/{num_of_holes}')
    return recon_im


def main():
    hole_im, input_img_path, eps, z, connectivity, out_path, plt = preprocess()
    holes_idx = get_holes_idx(hole_im)
    boundary_idx = get_boundary_idx(hole_im, connectivity)

    # initialize the weight func, you may replace to different func.
    # please note only the initializing the weight function is needed
    weight_class = Weight(z, eps)
    recon_im = fill_hole(hole_im, boundary_idx, holes_idx, weight_class)
    save_and_plot_image(recon_im, out_path, 224, plt)
    plt.savefig('final_results_connectivity_' + str(connectivity) + '.png')
    plt.show()



if __name__ == '__main__':
    main()
