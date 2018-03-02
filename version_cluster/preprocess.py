import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def to_gray(image):
    gray_image =  np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return gray_image

def preprocess(images):
    corp_image=np.zeros((220,168,4))
    for i in range(4):
        corp_image[0:210,0:160,i] = to_gray(images[i])
    ressampled_image = corp_image[::2,::2]
    img = np.expand_dims(ressampled_image[13:97], axis=0).astype(np.uint8)
    plt.imshow(img[0], cmap=cm.Greys_r)
    plt.show()
    return img

