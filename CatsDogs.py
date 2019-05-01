import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

datadir = os.getcwd()
categories = ["Parasitized", "Uninfected"]

training_data = []
image_size = 50
def create_training_data():
    global image_size
    for category in categories:
        path = os.path.join(datadir, category)   
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (image_size, image_size))
                plt.imshow(new_array)
                plt.show()
                if class_num==0:
                    training_data.append([new_array, [1,0]])
                else:
                    training_data.append([new_array, [0,1]])
            except:
                pass
        
    

create_training_data()

print(len(training_data))
#Initially, it is goiing to read all dog images and then all cat images.
#So initially it will have good accuracy till it switches from dog to cat and prediction goes wrong
import random
random.shuffle(training_data)
#First run for all then limit to 10
for sample in training_data[:10]:
    #0 is image array and 1 is whether its dog or cat
    print(sample[1])

#Two lists
    #One to store data of image and one more for labels
X = []
Y = []

for contents in training_data:
    X.append(contents[0])
    Y.append(contents[1])

#As of now, we cannot pass lists to Keras
#X has to be converted to numpy  array

# 1 is for grayscale, 3 for color image,
#-1 means how many features we have.
#If you put -1, it means catch all accepted
X = np.array(X).reshape(-1, image_size*image_size)
print(X[0].shape)

#Stage 4

import pickle

# We need to save model for tweaking later
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

