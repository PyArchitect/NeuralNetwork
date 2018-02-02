# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing CNN
classifier = Sequential()

# Step 1 Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64, 3), activation= 'relu'))

# Step 2 Max Pooling
classifier.add(MaxPooling2D(pool_size= (2, 2)))

# Step 3 Flattening
classifier.add(Flatten())

# Step 4 Full connection

# Hidden Layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#Output layer (sigmoid function - binary outcome, in case we would have had outcome with > than 2 categories softmax func)
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN (categorical cross entropy in case many labels, binary_cross_entropy - two labels)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Part 2 - Fitting the CNN to the images

# image augmentation - preprocess images to prevent overfitting (if we dont do it we might get great accuracy result on training set, but poor one on test set due to an overfit on the train set)
from keras.preprocessing.image import ImageDataGenerator

# train set
train_datagen = ImageDataGenerator(
        # rescale pixels of train set to values from [0,255] to [0,1]
        rescale=1./255,
        # random transvections
        shear_range=0.2,
        # random zoom
        zoom_range=0.2,
        # images will be flipped horizontally
        horizontal_flip=True)

# test set; rescale the images of test set  from [0,255] to [0,1]
test_datagen = ImageDataGenerator(rescale=1./255)

# Create training set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        # dimensions of images expected by CNN, above we reduced it to 64x64
        target_size=(64, 64),
        # number of images that will go through CNN after which the weights will be updated
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Fit CNN to training set and test its performance on test set
classifier.fit_generator(
        training_set,
        # number of images in a training set
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)












































