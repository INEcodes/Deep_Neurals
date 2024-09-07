import torch
from torchvision import models
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
#print(dir(models)) #which all the models are available for the torch models.

weights = ResNet101_Weights.DEFAULT
preprocess = weights.transforms()   #pre-processing so that the image is best suitable for the image for the model.
#print(preprocess)  # tells us that the pre process makes dimentions of image as 224 x 224, also has same mean, std.

img_path = 'Image_classification/Tahr.jpg'
img = Image.open(img_path)
resnet = resnet101(weights=weights)
#print(resnet)  # this is a conv2d model out of the all, has 171M models.
#print(img)  checking the image working.

batch = preprocess(img).unsqueeze(0)
#print(batch.shape)  #resizing of the image for the working as we need it to in the form of [1,3,224,224].

resnet.eval()
prediction = resnet(batch).squeeze(0).softmax(0)
#print(prediction.shape) #requires the size to be [1000].

class_id = prediction.argmax().item()
#print(class_id) # this gives us the output as 350 so how we will be changing this to a feasable output. this has more explanation to it.

score = prediction[class_id].item() # this gives us the index value of the 1000 item list of the tensor one and how close does it matches.

print(f'Class_id : {weights.meta["categories"][class_id]}, Score:{score}')
#Class_id : ibex, Score:0.7664992213249207, this is the output given to us from the above print

'''
This provides us the model to guess the animal from the image of the animal.

Here we see the output as ibex for a tahr, but this is an understanable error as they are quite similar.
'''















