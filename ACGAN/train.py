import tensorflow as tf
import cv2
import os
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, Reshape, LayerNormalization, LeakyReLU
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score

class ReadDataset:
    def __init__(self, dataset_path, image_shape):
        self.dataset_path = dataset_path
        self.image_shape = image_shape

    def read_images(self):
        images = []
        labels = []
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.resize(img, self.image_shape[:2])  # Resize image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0  

                images.append(img)
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

class Acgan:
    def __init__(self, eta, batch_size, epochs, weight_decay, latent_space,
                 image_shape, kernel_size, num_subclasses):
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.latent_space = latent_space
        self.image_shape = image_shape
        self.kernel_size = kernel_size
        self.num_subclasses = num_subclasses

    def data(self, images, labels):
        y_train = tf.keras.utils.to_categorical(labels, num_classes=self.num_subclasses)
        self.images = images
        self.labels = y_train

    def generator(self, inputs, labels):
        filters = [256, 128, 64, 32]
        padding = 'same'
        x = inputs
        y = labels
        x = layers.concatenate([x, y])
        x = layers.Dense(1024, )(x)
        x = layers.Dense(8*8*filters[0],
                         kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
        x = layers.Reshape((8, 8, filters[0]))(x)
        for filter in filters:
            if filter >= 64:
                strides = 2
            else:
                strides = 1
            x = LayerNormalization()(x)
            x = layers.Activation('relu')(x)
            x = Conv2DTranspose(filter, kernel_size=self.kernel_size, padding=padding,
                      strides=strides)(x)
        x = Conv2DTranspose(3, kernel_size=self.kernel_size, padding=padding)(x)
        x = layers.Activation('sigmoid')(x)
        y = Dense(self.num_subclasses, activation='softmax')(x)  
        self.generatorModel = models.Model(inputs=[inputs, labels],
                                           outputs=x,
                                           name='generator')

    def discriminator(self, inputs):
        x = inputs
        filters = [32, 64, 128, 256]
        padding = 'same'
        for filter in filters:
            if filter < 256:
                strides = 2
            else:
                strides = 1
            x = Conv2D(filter, kernel_size=self.kernel_size, padding=padding,
                    strides=strides,
                    kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x) 
        labels_output = Dense(self.num_subclasses, activation='softmax')(x)  
        self.discriminatorModel = models.Model(inputs=inputs,
                                            outputs=[outputs, labels_output],
                                            name='discriminator')

    def build(self):
        generator_input = layers.Input(shape=(self.latent_space,))
        discriminator_input = layers.Input(shape=(self.image_shape))
        labels_input = layers.Input(shape=(self.num_subclasses, ))  
        self.generator(generator_input, labels_input)
        self.discriminator(discriminator_input)
        G = self.generatorModel
        D = self.discriminatorModel
        D.compile(loss=['mse', 'categorical_crossentropy'],
                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.eta,
                                                        weight_decay=self.weight_decay))
        D.summary()
        G.summary()
        D.trainable = False
        GAN = models.Model(inputs=[generator_input, labels_input],
                           outputs=D(G([generator_input, labels_input])))
        GAN.compile(loss=['mse', 'categorical_crossentropy'],
                   optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.eta*0.5,
                                                          weight_decay=self.weight_decay*0.5))
        GAN.summary()
        return G, D, GAN
    
    def trainAlgorithm(self, G, D, GAN):
        for epoch in range(self.epochs):
            indexs = np.random.randint(0, len(self.images), size=(self.batch_size, ))
            realImages = self.images[indexs]
            realLabels = self.labels[indexs]
            realTag = tf.ones(shape=(self.batch_size, ))
            noize = tf.random.uniform(shape=(self.batch_size,
                                              self.latent_space), minval=-1,
                                     maxval=1)
            fakeLabels = tf.keras.utils.to_categorical(np.random.choice(range(self.num_subclasses), size=(self.batch_size,)),
                                                      num_classes=self.num_subclasses)
            fakeImages = G.predict([noize, fakeLabels])
            fakeTag = tf.zeros(shape=(self.batch_size,))
            discriminatorInput = np.concatenate((realImages, fakeImages), axis=0)
            discriminatorLabels = np.concatenate((realLabels, fakeLabels), axis=0)
            discriminatorTag = np.concatenate((realTag, fakeTag), axis=0)
            discriminatorLoss = D.train_on_batch(discriminatorInput,
                                                 [discriminatorTag, discriminatorLabels])
            noize = tf.random.uniform(shape=(self.batch_size,
                                              self.latent_space), minval=-1,
                                     maxval=1)
            generatorLabels = tf.keras.utils.to_categorical(np.random.choice(range(self.num_subclasses), size=(self.batch_size,)),
                                                             num_classes=self.num_subclasses)
            ganTag = tf.ones(shape=(self.batch_size,))
            ganLoss = GAN.train_on_batch([noize, generatorLabels], ganTag)
            print("epoch:", epoch, "discriminator loss:", discriminatorLoss,
                  "GAN loss:", ganLoss)


dataset_path = "F:/Akash-GMDfVPs/Updated/6chest-disease/chest-xray-with-multi-disease/train-20230326T152931Z-001/train"
image_shape = (64, 64, 3)
num_subclasses = 6 
readDatasetObject = ReadDataset(dataset_path, image_shape)
images, labels = readDatasetObject.read_images()

unique_labels = np.unique(labels)
label_dict = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_dict[label] for label in labels])


acgan = Acgan(eta=0.0001, batch_size=32, epochs=32000, weight_decay=6e-9,
              latent_space=100, image_shape=(64, 64, 3), kernel_size=5,
              num_subclasses=num_subclasses)
acgan.data(images, labels)
G, D, GAN = acgan.build()
acgan.trainAlgorithm(G, D, GAN)


G.save('F:/Akash-GMDfVPs/Ref/PES-University-Capstone-Project-main/Ak-Fa/AC_user/generator_6subclasses.h5')


G = tf.keras.models.load_model('F:/Akash-GMDfVPs/Ref/PES-University-Capstone-Project-main/Ak-Fa/AC_user/generator_6subclasses.h5')


user_label = input("Enter label specification: ") 


noize = tf.random.uniform(shape=(1, 100), minval=-1, maxval=1)
user_label_idx = label_dict[user_label]
label_vector = tf.keras.utils.to_categorical(user_label_idx, num_classes=len(unique_labels)).reshape(1, -1)
generated_image = G.predict([noize, label_vector])


plt.imshow(generated_image[0])
plt.title(user_label)
plt.show()
