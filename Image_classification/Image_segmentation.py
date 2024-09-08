from skimage.io import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

tensor = imread(r'Image_classification\Tahr.jpg')
tensor = tensor / 255.0
image = tensor.reshape(-1, 3)

kmeans = KMeans(n_clusters=2)#changing the value here allows us to access the depth and the depth of the image.
kmeans.fit(image)

seg_img = kmeans.cluster_centers_[kmeans.labels_]
seg_img = seg_img.reshape(tensor.shape)

plt.imshow(seg_img)
plt.show()


#this divides the image in two or n colour of the cluster. can be useful for depth analysis and the image or rays tracing.

