import cv2
from numpy import asarray
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt 

#converting the image into a matrix form:
def capture():
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Capture a single frame from the camera
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture image.")
        exit()

    # Convert the captured frame (which is a NumPy array) into a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert the PIL image to a NumPy array
    numpydata = asarray(img)

    # Data
    print(numpydata)

    # Optionally, display the captured image
    img.show()

capture()
def colorspace():
    cap1 = cv2.VideoCapture(0)
    ret, im = cap1.read()  # Read frame from camera
    cap1.release()  # Release the camera

    if not ret:
        print("Failed to capture image")
        return

    # Convert BGR (OpenCV default) to RGB for processing
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Convert RGB image to LAB color space
    im1 = rgb2lab(im)

    # Set A and B channels to 0 to make the image grayscale
    im1[..., 1] = 0
    im1[..., 2] = 0

    # Convert back to RGB from LAB
    im1 = lab2rgb(im1)

    # Display the original and grayscale images
    plt.figure(figsize=(12, 6))  # Adjusting the figure size for better display
    plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Original image', size=20)
    plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('Grayscale image', size=20)
    plt.show()
    print("Done coverting the image colors")

colorspace()