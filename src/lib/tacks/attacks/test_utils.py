import matplotlib.cbook as cbook
import matplotlib.pyplot as plt

from tacks.attacks.utils import Patch
from tacks.utils.image import (
    opencv_to_torch_img,
    torch_to_opencv_img,
    permute_axes_img,
)

positive_patches = [
    {
        'shape': 'rectangle',
        'topleft': (27, 80),
        'bottomright': (
            50,
            100,
        ),
    },
    {
        'shape': 'circle',
        'centre': (64, 64),
        'radius': 10,
    },
    {
        'shape': 'ellipse',
        'centre': (100, 32),
        'major_axis_length': 6,
        'minor_axis_length': 15,
    },
]

negative_patches = [{'shape': 'circle', 'centre': (64, 64), 'radius': 4}]

# load the image
with cbook.get_sample_data('Minduka_Present_Blue_Pack.png') as image_file:
    np_img = plt.imread(image_file)[..., 0:3]

img_size = np_img.shape[0:2]

img = opencv_to_torch_img(np_img)
img_rgb = img.clone()
img_rgba = img.clone()
img_bw = img.clone()
img_bwa = img.clone()

# RGB mask
rgb_mask = Patch(size_px=img_size, size_mm=(200, 200), cell_size=2)
_ = [rgb_mask.add_patch(patch) for patch in positive_patches]
_ = [rgb_mask.add_patch(patch, negative=True) for patch in negative_patches]
rgb_mask(img_rgb)
rgb_mask.get_pdf('test.pdf')

# RGBA mask
rgba_mask = Patch(size_px=img_size, size_mm=(200, 200), is_transparent=True, cell_size=4)
_ = [rgba_mask.add_patch(patch) for patch in positive_patches]
_ = [rgba_mask.add_patch(patch, negative=True) for patch in negative_patches]
rgba_mask(img_rgba)

# BW mask
bw_mask = Patch(size_px=img_size, size_mm=(200, 200), is_bw=True, cell_size=16)
_ = [bw_mask.add_patch(patch) for patch in positive_patches]
_ = [bw_mask.add_patch(patch, negative=True) for patch in negative_patches]
bw_mask(img_bw)


# BWA mask
bwa_mask = Patch(
    size_px=img_size, size_mm=(200, 200), is_bw=True, is_transparent=True, cell_size=32
)
_ = [bwa_mask.add_patch(patch) for patch in positive_patches]
_ = [bwa_mask.add_patch(patch, negative=True) for patch in negative_patches]
bwa_mask(img_bwa)


plt.figure()

# RGB
plt.subplot(4, 3, 1)
plt.imshow(permute_axes_img(img_rgb, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBG - Image')

plt.subplot(4, 3, 2)
plt.imshow(permute_axes_img(rgb_mask.values, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBG - Values')

plt.subplot(4, 3, 3)
plt.imshow(rgb_mask.mask)
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBG - Image')

# RGBA
plt.subplot(4, 3, 4)
plt.imshow(permute_axes_img(img_rgba, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBGA - Image')

plt.subplot(4, 3, 5)
plt.imshow(permute_axes_img(rgba_mask.values, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBGA - Values')

plt.subplot(4, 3, 6)
plt.imshow(rgba_mask.mask)
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBG - Image')

# WB
plt.subplot(4, 3, 7)
plt.imshow(permute_axes_img(img_bw, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBGA - Image')

plt.subplot(4, 3, 8)
plt.imshow(permute_axes_img(bw_mask.values, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('BW - Values')

plt.subplot(4, 3, 9)
plt.imshow(bw_mask.mask)
plt.xticks([], [])
plt.yticks([], [])
plt.title('BW - Image')


# WBA
plt.subplot(4, 3, 10)
plt.imshow(permute_axes_img(img_bwa, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('RBGA - Image')

plt.subplot(4, 3, 11)
plt.imshow(permute_axes_img(bwa_mask.values, 'WHC'))
plt.xticks([], [])
plt.yticks([], [])
plt.title('BWA - Values')

plt.subplot(4, 3, 12)
plt.imshow(bwa_mask.mask)
plt.xticks([], [])
plt.yticks([], [])
plt.title('BWA - Image')

plt.show()
