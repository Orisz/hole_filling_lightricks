import argparse
from PIL import Image
import numpy as np


def parse_args_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', help='power of the norm', type=int, default=1)
    parser.add_argument('--input_img', help='input image path', default='Input/lena.jpg')
    parser.add_argument('--input_mask', help='mask path', default='Input/lena_mask.png')
    parser.add_argument('--eps', help='epsilon', type=float, default=1e-3)
    parser.add_argument('--connectivity', help='type 4 or 8', type=int, default=4)
    parser.add_argument('--out_img', default='.', help='folder to output image')
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
    im.show()
    mask.show()
    im_np, mask_np = rgb_2_gray_np(im, mask)
    binary_mask = convert_mask_2_binary(mask_np)
    im_np = normalize_image(im_np)
    # hole_im = im_np * binary_mask #for plots
    hole_im = im_np
    hole_im[binary_mask == 0] = -1
    # print(f'im max:{np.amax(im_np)}, min:{np.amin(im_np)}, mask max{np.amax(binary_mask)}, mask min:{np.amin(binary_mask)}')
    # print(f'hole_im max:{np.amax(hole_im)}, hole_im min:{np.amin(hole_im)}')
    return hole_im, input_img_path, eps, z, connectivity, out_path


def save_hole_image(hole_im):
    img = (hole_im*255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save('Input/hole_im_lena_2.jpg')
    img.show()


def get_boundary_idx(hole_im, connectivity):
    boundary_idx = []
    if connectivity == 4:
        neighbours_idx = lambda x, y: [[x-1, y], [x+1, y], [x, y-1], [x, y+1]]
    # connectivity == 8
    else:
        neighbours_idx = lambda x, y: [[x-1, y], [x+1, y], [x, y-1], [x, y+1],
                                       [x+1, y+1], [x+1, y-1], [x-1, y+1], [x-1, y-1]]
    for y in hole_im.shape[0]:
        for x in hole_im.shape[1]:
            neighbours = hole_im(neighbours_idx(x, y))
            cond = np.all(neighbours == neighbours[0, 0])
            if not cond:
                boundary_idx.append([y, x])
    return boundary_idx


def get_holes_idx(hole_im):
    idx = hole_im == -1
    return idx


def main():
    hole_im, input_img_path, eps, z, connectivity, out_path = preprocess()
    # save_hole_image(hole_im)
    # print(f'im max:{np.amax(m_np)}, min:{np.amin(im_np)}, mask max{np.amax(mask_np)}, mask min:{np.amin(mask_np)}')
    holes_idx = get_holes_idx(hole_im)
    boundary_idx = get_boundary_idx(hole_im, connectivity)


if __name__ == '__main__':
    main()
