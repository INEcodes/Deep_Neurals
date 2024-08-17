import matplotlib.pyplot as plt
from torchvision.io import read_image


mask = read_image("Object_detection_torchvision/PennFudanPed/PedMasks/FudanPed00046_mask.png")
image = read_image("Object_detection_torchvision/PennFudanPed/PNGImages/FudanPed00046.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
plt.show()