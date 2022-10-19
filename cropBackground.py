#/usr/bin/env python3

import nibabel as nib
import numpy as np
import sys
import tqdm

def stripSkull(image):
    contour = 4
    data = image.get_data()
    minVal = data[0,0,0]
    threshold = minVal+0.001
    foreground = data > threshold
    (x,) = np.nonzero(np.amax(foreground, axis=(1,2)))
    (y,) = np.nonzero(np.amax(foreground, axis=(0,2)))
    (z,) = np.nonzero(np.amax(foreground, axis=(0,1)))
    affine = image.affine.copy()
    start = (x.min() - contour, y.min() - contour, z.min() - contour)
    stop = (x.max()+contour+1, y.max()+contour+1, z.max()+contour+1)
    affine[0:3,3] = nib.affines.apply_affine(affine.copy(), start)
    image = nib.Nifti1Image( \
            data[start[0]:stop[0],start[1]:stop[1],start[2]:stop[2]], affine)
    return image

if __name__ == '__main__':
    if len(sys.argv) == 3:
        img = nib.load(sys.argv[1])
        img = stripSkull(img)
        nib.save(img, sys.argv[2])
    else:
        for line in tqdm.tqdm(sys.stdin.readlines()):
            imgIn, imgOut = line.strip().split(',')
            img = nib.load(imgIn)
            img = stripSkull(img)
            print(img.shape)
            nib.save(img, imgOut)

