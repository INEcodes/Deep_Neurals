import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#this directly converts this into grayscale without focusing on geometric transformations without the complexity of color channels.
img = rgb2gray(imread('Image_classification\Tahr.jpg'))
w, h = img.shape
transformed_images = []
labels = []
transformed_images.append(img)
labels.append('1. Original')

#Applying Identity transformers
mat_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
img1 = ndi.affine_transform(img, mat_identity)
mat_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
img1 = ndi.affine_transform(img, mat_identity)
transformed_images.append(img1)
labels.append('2. Identity Transform')

#Applying Reflection Transform (Along the X-Axis):
mat_reflect = np.array([[1,0,0],[0,-1,0],[0,0,1]]) @ np.array([[1,0,0],[0,1,-h],[0,0,1]])
img2 = ndi.affine_transform(img, mat_reflect) #offset=(0,h)
transformed_images.append(img2)
labels.append('3. Reflection Transform')

#Scale the Image:
s_x, s_y = 0.75, 1.25
mat_scale = np.array([[s_x,0,0],[0,s_y,0],[0,0,1]])
img3 = ndi.affine_transform(img, mat_scale)
transformed_images.append(img3)
labels.append('4. Scale the Image')

#Rotate the Image by 30degree Counter-Clockwise:
theta = np.pi / 6
mat_rotate = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
img4 = ndi.affine_transform(img1,mat_rotate)
transformed_images.append(img4)
labels.append('5. Rotate the Image')


# Apply Shear Transform to the Image:
lambda1 = 0.5
mat_shear = np.array([[1, lambda1, 0], [lambda1,1,0],[0,0,1]])
img5 = ndi.affine_transform(img1,mat_shear)
transformed_images.append(img5)
labels.append('6. Shear Transform')

#combine all transformations:
mat_all = mat_identity @ mat_reflect @ mat_scale @ mat_rotate @ mat_shear
img_final = ndi.affine_transform(img, mat_all)

#plot all transformations:
fig, axes = plt.subplots(1, len(transformed_images), figsize=(20, 10))
for ax, image, label in zip(axes, transformed_images, labels):
    ax.imshow(image, cmap='gray')
    ax.set_title(label)
    ax.axis('off')
    plt.tight_layout()
plt.show()