import numpy as np
from PIL import Image, ImageFilter

image = np.array(Image.open("/home/sang/cifar/test/" + "7277_deer.png"))
image = image.reshape((3 * 32 * 32, ))
image = image.astype(np.float64) / 255
np.savetxt('test.out', image, delimiter=',')