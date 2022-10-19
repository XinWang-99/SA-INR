import skimage.util
import tqdm
import math
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import crop
from scipy.fft import fft2, fftshift
from einops import rearrange
import SimpleITK as sitk
import os
import pandas as pd

class SpectralSharpness:
    def __init__(self, block_size_s1: int, overlap_s1: float,  block_size_s2: int, overlap_s2: float,
                 verbose: bool = False, border_padding: int = 8):
        # if overlap < 0 or overlap >= 1:
        #     raise ValueError('The overlap ratio must be a float number between 0 and 1.')
        self.block_size_s1 = (block_size_s1, block_size_s1)
        self.block_size_s2 = (block_size_s2, block_size_s2)
        self.overlap_s1 = overlap_s1
        self.overlap_s2 = overlap_s2
        self.verbose = verbose
        self.padding = border_padding

    def start_points(self, size: int, split_size: int, overlap: float = 0.0):
        """
        starting points of overlapping block generation
        :param size: image size, int
        :param split_size: image block size, int
        :param overlap: overlap portion, float between 0-1
        :return:
        """
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    def generate_image_blocks(self, image, block_size, overlap):
        """
        generate image blocks with certain block size and overlapping ratio
        :param image: np.array, image to be split
        :param block_size: int, spatial length of the block
        :param overlap: float, overlapping ratio
        :return: list of image blocks
        """
        height, width = image.shape
        split_height, split_width = block_size
        block_list = []
        X_points = self.start_points(width, split_width, overlap)
        Y_points = self.start_points(height, split_height, overlap)
        for i in Y_points:
            for j in X_points:
                split = image[i:i + split_height, j:j + split_width]
                block_list.append(split)
        return block_list

    def block2image(self, image_blocks, image_size, block_size, overlap):
        """
        stack a list of image blocks into a full image
        :param image_blocks: list of image blocks
        :param image_size: target spatial size of the reconstructed image
        :param block_size: spatial size of the image blocks
        :param overlap: overlapping ratio of spatial blocks
        :return: numpy.array, reconstructed image
        """
        height, width = image_size
        image = np.zeros(image_size)
        split_height, split_width = block_size
        X_points = self.start_points(width, split_width, overlap)
        Y_points = self.start_points(height, split_height, overlap)
        index = 0
        for i in Y_points:
            for j in X_points:
                image[i:i + split_height, j:j + split_width] = image_blocks[index]
                index += 1
        return image

    def contrast_map(self, image_block):
        image_block = image_block * 255.0
        block_lum = (0.7656 + 0.0364 * image_block) ** 2.2
        mean_val = block_lum.mean()
        range_val = block_lum.max() - block_lum.min()
        if mean_val > 2 and range_val > 5:
            return True
        else:
            return False

    def get_magnitude_spectrum(self, xft):
        """
        get the magnitude spectrum across all orientations
        :param xft: shifted 2D DFT of image block x
        :return: the magnitude spectrum across all orientations
        """
        assert xft.shape[0] == xft.shape[1], 'We expect square image blocks'
        radius = xft.shape[0]
        radius_mask = np.zeros(xft.shape)
        radius_mask[xft.shape[0] // 2, :] = np.ones(radius)
        magnitude_plot = 0.0
        for rotate in range(360):
            rotate = rotate / 180 * math.pi
            ty = xft.shape[0] // 2
            tx = xft.shape[1] // 2
            # rotate matrix
            T = np.array([[math.cos(rotate), -math.sin(rotate), (1 - math.cos(rotate)) * tx + math.sin(rotate) * ty],
                          [math.sin(rotate), math.cos(rotate), (1 - math.cos(rotate)) * ty - math.sin(rotate) * tx],
                          [0, 0, 1]])
            # warp rotation with rotate matrix
            radius_mask_rotated = cv2.warpAffine(radius_mask, T[:2, :].astype(np.float32), xft.shape)
            masked_xft = xft * radius_mask_rotated
            # reversed rotation
            T_r = np.array(
                [[math.cos(-rotate), -math.sin(-rotate), (1 - math.cos(-rotate)) * tx + math.sin(-rotate) * ty],
                 [math.sin(-rotate), math.cos(-rotate), (1 - math.cos(-rotate)) * ty - math.sin(-rotate) * tx],
                 [0, 0, 1]])
            recovered_xft = cv2.warpAffine(masked_xft, T_r[:2, :].astype(np.float32), xft.shape)
            magnitude_plot += recovered_xft
        magnitude_spectrum = magnitude_plot[xft.shape[0] // 2, :]
        half_spectrum = magnitude_spectrum[np.argmax(magnitude_spectrum):]
        return half_spectrum

    def get_coeff(self, magnitude_spectrum):
        """
        linear interpolation for calculating alpha
        :param magnitude_spectrum:
        :return:
        """
        num_steps = len(magnitude_spectrum)
        x = np.linspace(0, 0.5, num_steps)[1:]
        p = np.polyfit(np.log(x), np.log(magnitude_spectrum)[1:], 1)
        alpha = -p[0]
        return alpha

    def TV(self,s_block):
        tv=0.25 * (abs(s_block[0] - s_block[1]) + abs(s_block[0] - s_block[2]) +
                       abs(s_block[1] - s_block[3]) + abs(s_block[2] - s_block[3]))
        #print(s_block,tv)
        return tv


    def __call__(self, image: np.array):
        """
        generate the s1 spectral sharpness map
        :param image: 2D RGB or grayscale image with shape (H, W, (C))
        :return: s1 map of the image
        """
        img = np.pad(image, pad_width=((self.padding, self.padding), (self.padding, self.padding)),
                     mode='reflect')
        #print(img.shape)
        image_blocks_s1 = self.generate_image_blocks(img, self.block_size_s1, self.overlap_s1)
        if self.verbose:
            image_blocks_s1 = tqdm.tqdm(image_blocks_s1)
        s1_map_blocks = []

        for block in image_blocks_s1:
            if self.contrast_map(block):
                bfs = np.abs(fftshift(fft2(block)))
                magnitude_spectrum = self.get_magnitude_spectrum(bfs)
                alpha = self.get_coeff(magnitude_spectrum)
                s1 = 1 - 1 / (1 + np.exp(-3 * (alpha - 2)))
            else:
                s1 = 0.0
            s1_map = np.zeros(block.shape)
            s1_map.fill(s1)
            s1_map_blocks.append(s1_map)
        s1_image = self.block2image(s1_map_blocks, img.shape, self.block_size_s1, self.overlap_s1)
        s1_image = crop(s1_image, crop_width=((self.padding, self.padding), (self.padding, self.padding)))

        image_blocks_s2 = self.generate_image_blocks(img, self.block_size_s2, self.overlap_s2)
        if self.verbose:
            image_blocks_s2 = tqdm.tqdm(image_blocks_s2)
        s2_map_blocks = []

        for block in image_blocks_s2:
            sub_blocks=self.generate_image_blocks(block, (2,2), 0.5)
            tv_list = []
            for s_block in sub_blocks:
                tv_list.append(self.TV(s_block.flatten()))
            s2=max(tv_list)
            s2_map = np.zeros(block.shape)
            s2_map.fill(s2)
            s2_map_blocks.append(s2_map)
        s2_image = self.block2image(s2_map_blocks, img.shape, self.block_size_s2, self.overlap_s2)
        s2_image = crop(s2_image, crop_width=((self.padding, self.padding), (self.padding, self.padding)))


        s3_image=(s1_image**0.5)*(s2_image**0.5)
        sorted_array = np.sort(s3_image.flatten())
        m = img.shape[0]
        n = img.shape[1]
        num = math.ceil(m * n * 0.01)
        s3=np.mean(sorted_array[-num:])
        return s1_image,s2_image,s3_image,s3

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--model_path',type=str,required=True)
    parser.add_argument('--scale',nargs='+', type=int,required=True)
    args = parser.parse_args()

    model=args.model_path
    scale=args.scale
    nii_dir="/hpc/data/home/bme/v-wangxin/vit/new_ep/{}/x{}x{}x{}/".format(model,scale[0],scale[1],scale[2])
    save_path="/hpc/data/home/bme/v-wangxin/vit/new_ep/{}/s3_x{}x{}x{}.csv".format(model,scale[0],scale[1],scale[2])
    evaluator = SpectralSharpness(64, 1/4, 8,1/2, True, 8)
    s3_list=[]
    j=0
    for nii in os.listdir(nii_dir):
        j+=1
        path=os.path.join(nii_dir,nii,'sr.nii.gz')
        img=sitk.GetArrayFromImage(sitk.ReadImage(path))
        s3_list_sub=[]
        for i in range(img.shape[0]):
            s1_map, s2_map, s3_map, s3 = evaluator(img[i,:,:])
            s3_list_sub.append(s3)
        for i in range(img.shape[1]):
            s1_map, s2_map, s3_map, s3 = evaluator(img[:, i, :])
            s3_list_sub.append(s3)
        for i in range(img.shape[2]):
            s1_map, s2_map, s3_map, s3 = evaluator(img[:, :, i])
            s3_list_sub.append(s3)
        s3_list.append(sum(s3_list_sub)/len(s3_list_sub))
        df=pd.DataFrame({'s3':s3_list},index=range(1,j+1))
        df.to_csv(save_path)
        #exit(0)