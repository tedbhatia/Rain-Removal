from __future__ import division, print_function, absolute_import
from sklearn.metrics import mean_squared_error

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from skimage.io import imread
from PIL import Image
from layer import *

class Derain:

    def __init__(self):
        """
            No need to change the path for testing, training or checkpoints. Simply place the
            test images (concatenated version) in the test folder.
            main is defined in skeleton.py
            Simply place the test images in the folder and run skeleton.py
        """
        # Training Params
        self.checkpoint_path = os.getcwd() + "/checkpoints/modelMSE_NP_XI_RS_LD500-0.1_5kLOAD-3000"
        self.batch_size = 8

        # Network Params
        self.learning_rate = 0.001
        self.training_steps = 0

        # Training Inputs
        self.training_path = os.getcwd() + "/training/"
        self.rain_image = []
        self.no_rain_image = []

        # Test Inputs
        self.test_path = os.getcwd() + "/test/"
        self.test_image = []
        self.test_truth_image = []
        self.test_set_size = 0

        L = Layers()

        #Loading rain images training data
        dirr = os.chdir(self.training_path + "rain")
        for filename in os.listdir(dirr):
            if filename.endswith(".jpg"):
                x = imread(filename)
                self.rain_image.append(x)

        #Loading no rain (ground truth) images training data
        dirr = os.chdir(self.training_path + "norain")
        for filename in os.listdir(dirr):
            if filename.endswith(".jpg"):
                x = imread(filename)
                self.no_rain_image.append(x)

        #Loading test data
        dirr = os.chdir(self.test_path)
        for filename in os.listdir(dirr):
            if filename.endswith(".jpg"):

                self.test_set_size += 1;
                imOb = Image.open(filename)
                w, h = imOb.size

                #Cropping the rain part of test
                area = (w/2, 0, w, h)
                RainImg = imOb.crop(area)
                RainImg = RainImg.resize((512, 512), Image.ANTIALIAS)
                self.test_image.append(np.asarray(RainImg))

                #Cropping the ground truth part of test
                area = (0, 0, w/2, h)
                noRainImg = imOb.crop(area)
                noRainImg = noRainImg.resize((512, 512), Image.ANTIALIAS)
                self.test_truth_image.append(np.asarray(noRainImg))


        self.rain_image_input = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
        self.no_rain_image_input = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
        self.globalstep = tf.placeholder(tf.int32)

        self.de_rained_image = L.autoenc(self.rain_image_input)

        #Optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #Learning Rate Decay
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step = self.globalstep, decay_steps=500, decay_rate=0.1, staircase=True)

        #Generating Variables
        auto_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AutoEncoder')

        #Loss
        self.Loss = tf.losses.mean_squared_error(self.no_rain_image_input,self.de_rained_image)

        #Minimizing function
        self.train_tot = optimizer.minimize(self.Loss, var_list=auto_vars)

    def train(self):
        """
            Trains the model on data given in path/train.csv
                which conatins the RGB values of each pixel of the image

            No return expected
        """
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            self.sess.run(init)

            #Loading the checkpoint
            self.load_model()
            X = []
            Y = []
            for i in range(1, self.training_steps+1):

                # To obtain a random mini batch
                start = np.random.randint(0, 700 - self.batch_size)
                end = start + self.batch_size

                if start>end:
                    start=0
                    end=self.batch_size

                batch_x= self.no_rain_image[start:end]
                z = self.rain_image[start:end]

                feed_dict = {self.no_rain_image_input: batch_x, self.rain_image_input: z, self.globalstep: i}
                _, tl, lr = self.sess.run([self.train_tot, self.Loss, self.learning_rate],
                                feed_dict=feed_dict)

                #Plotting the loss fucntion
                #X.append(i)
                #Y.append(tl)

            # Saving the trained weights model
            #self.save_model(i)

            #Plotting the derained vs original image
            #f, a = plt.subplots(3)

            psnrSum = 0
            for i in range(0, self.test_set_size) :
                z = self.test_image[i:i+1]
                z_truth = self.test_truth_image[i:i+1]
                g = self.sess.run(self.de_rained_image, feed_dict={self.rain_image_input: z})

                #De rained image
                img = np.reshape(g, newshape=(512, 512, 3))
                #Rainy test image
                img1 = np.reshape(z, newshape=(512, 512, 3))
                #Ground truth image
                img2 = np.reshape(z_truth, newshape=(512, 512, 3))

                #Calculating PSNR of individual image
                psnrRes = self.psnr(img2, img)
                print("PSNR %i is %f" %(i+1, psnrRes))
                psnrSum += psnrRes

                # Showing the individual de rained vs rainy image along with loss function
                #a[0].imshow(img.astype(np.uint8))
                #a[1].imshow(img1.astype(np.uint8))
                #a[2].plot(X,Y)
                #plt.show()

            print("PSNR Average : %f" %(psnrSum/self.test_set_size))

    def save_model(self, step):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.path+self.filename, global_step=step)
        print("Model saved in file: %s" % save_path)


    def load_model(self):
        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint_path)
        print("Model restored.")


    def psnr(self, originalImage, predictedImage):
        mse = mean_squared_error(originalImage.reshape(-1,1), predictedImage.reshape(-1,1))
        maximum = 255
        return 20*math.log10(maximum) - 10*math.log10(mse)


if __name__ == '__main__':

    AutoEncoder = Derain()
    AutoEncoder.train()
