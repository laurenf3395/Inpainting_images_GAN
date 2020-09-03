from datetime import datetime
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models
from model import Model
from PIL import Image
#from PIL.Image import paste
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:

    def __init__(
        self, model, train_dataset, val_dataset, test_dataset,
        out_path='/cluster/scratch/laurenf/HW6/Handout/dataset/Results'
        ):
        self.use_adv = model.use_adv

        #UNCOMMENT BELOW LINE FOR ADVERSARIAL LOSS AND COMMENT LINE 26
        #self.model,self.discriminator_model,self.generator_model = model.build_model()
        self.model = model.build_model()#  #combined
        self.sess = keras.backend.get_session()
        
        # Separate member for the tf.data type
        self.train_dataset = train_dataset
        self.train_dataset_tf = self.train_dataset.dataset_tf

        self.val_dataset = val_dataset
        self.val_dataset_tf = self.val_dataset.dataset_tf

        self.test_dataset = test_dataset
        self.test_dataset_tf = self.test_dataset.dataset_tf

        self.callbacks = []
        self.predicted = None
        # Output path
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)


    def train(self, nepochs=5):
        self.create_callbacks()

        self.callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.show_prediction(epoch)
            )
        )

        if not self.use_adv:
            self.model.fit(
                self.train_dataset_tf, 
                epochs=nepochs, 
                steps_per_epoch=7*self.train_dataset.num_batches,
                callbacks=self.callbacks,
                validation_data=self.val_dataset_tf,
                validation_steps=self.val_dataset.num_batches,
            )

        else:
            self.train_adv_loss(nepochs)

    
    def create_callbacks(self):
        """ Saves a  tensorboard with evolution of the loss """

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir, 
            write_graph=True
        )

        self.callbacks.append(tensorboard_callback)
        
        
    def show_prediction(self, epoch):
        """ Save prediction image every 25 epochs """
        if epoch % 25 == 0:
            # Use the model to predict the values from the training dataset.
            train_predictions = self.run_prediction(self.train_dataset_tf)
            val_predictions = self.run_prediction(self.val_dataset_tf)

            fig=plt.figure()

            fig.add_subplot(2, 3, 1)
            plt.imshow(train_predictions[0])
            plt.axis('off')

            fig.add_subplot(2, 3, 2)
            plt.imshow(train_predictions[1])
            plt.axis('off')

            fig.add_subplot(2, 3, 3)
            plt.imshow(train_predictions[2])
            plt.axis('off')

            fig.add_subplot(2, 3, 4)
            plt.imshow(val_predictions[0])
            plt.axis('off')

            fig.add_subplot(2, 3, 5)
            plt.imshow(val_predictions[1])
            plt.axis('off')

            fig.add_subplot(2, 3, 6)
            plt.imshow(val_predictions[2])
            plt.axis('off')

            plt.savefig(os.path.join(
                self.out_path,"res_epoch_disc_{:03d}.png".format(epoch)
            ))
            plt.close()

    def test(self,steps):
        results = self.evaluate(self.test_dataset_tf,steps)

        if steps % 5 == 0:
            test_predictions = self.run_prediction_test(self.test_dataset_tf)

            fig1 = plt.figure()

            fig1.add_subplot(1, 3, 1)
            plt.imshow(test_predictions[0])
            plt.axis('off')

            fig1.add_subplot(1, 3, 2)
            plt.imshow(test_predictions[1])
            plt.axis('off')

            fig1.add_subplot(1, 3, 3)
            plt.imshow(test_predictions[2])
            plt.axis('off')

            plt.savefig(os.path.join(
                self.out_path, "test_results_step_{:03d}.png".format(steps)
            ))
            plt.close()

    def convert_to_img(self, np_array):
        img = np_array.copy()
        img *= 255
        img = np.maximum(np.minimum(255, img), 0)
        img = img.astype(np.uint8)

        return img


    def run_prediction(self, dataset):
        # Get a input, groundtruth pair from a tf.dataset
        input_img_array, gt_img_array = self.sess.run(
            dataset.make_one_shot_iterator().get_next()
        )

        return self.create_full_images(input_img_array, gt_img_array)

    def run_prediction_test(self, dataset):
        # Get a input, groundtruth pair from a tf.dataset
        input_img_array, gt_img_array = self.sess.run(
            dataset.make_one_shot_iterator().get_next()
        )
        return self.generate_test_images(input_img_array,gt_img_array)

    def create_full_images(self, input_img_array, gt_img_array):
        """ 
        Create full images for visualization for one input image
    
        Input
           * input_img_array: input_img with cropped region, as (floats)
                - size: (batch_size, 128, 128, 3)
           * gt_img_array: groundtruth corresponding to the cropped region (floats)
                - size: (batch_size, 64, 64, 3)

        Output
           * input_img: input_img with cropped region, as numpy array (uint8)
                - size: (128, 128, 3)f
           * full_gt: input_img with with cropped region replaced by gt (uint8)
                - size: (128, 128, 3)
           * full_pred: input_img with with cropped region replaced by prediction
                - size: (128, 128, 3)
        """
        #TODO 3.1#NEED TO DO!!!!!
        #input_img = np.zeros([128, 128, 3])
        #full_gt   = np.zeros([128, 128, 3])
        #full_pred = np.zeros([128, 128, 3])

        input_img = input_img_array[0,:,:,:] * 255
        input_img = input_img.astype('uint8')

        fig2 = plt.figure()

        fig2.add_subplot(1, 1, 1)
        plt.imshow(input_img)
        plt.axis('off')
        plt.savefig("res_epoch_img.png")
        plt.close()
        inp_cpy = np.copy(input_img)

        gt_img = gt_img_array[0,:,:,:]*255 #(64,64,3)
        gt_img = gt_img.astype('uint8')
        fig3 = plt.figure()

        fig3.add_subplot(1, 1, 1)
        plt.imshow(gt_img)
        plt.axis('off')
        plt.savefig("res_epoch_gt_img.png")
        plt.close()
        full_gt = inp_cpy #uint8 type
        #full_gt[32:96,32:96,:] = gt_img
        #h = full_gt.shape[0] #128
        #w = full_gt.shape[1] #128
        #print("H")
        #print(h)
        for y in range(7,70):
            for x in range(7,70):
                v = full_gt[y:y+50,x:x+50,:]
                #print("V")
                #print(v.shape)
                if np.all(v==0):
                    y_rand = y
                    x_rand = x
                    break


        full_gt[y_rand-7:y_rand+57,x_rand-7:x_rand+57,:] = gt_img
        full_gt = full_gt.astype('uint8')

        #pred = self.generator_model.predict(input_img_array)
        pred = self.model.predict(input_img_array)
        #ADVERSARIAL
        #pred = self.predicted
        full_pred = inp_cpy
        pred_1 = pred[0, :, :, :] * 255

        pred_1 = pred_1.astype('uint8')
        full_pred[y_rand-7:y_rand+57,x_rand-7:x_rand+57,:] = pred_1
        #full_pred = full_pred.astype('uint8')
        return input_img, full_gt, full_pred

    
    def evaluate(self, dataset,steps):
        #TODO Task 3.3
        scores = self.model.evaluate(dataset, batch_size=5,steps=steps)
        print('test loss, test acc:', scores)
        f = open(os.path.join(self.out_path, "test_scores.txt"), "a+")
        f.write("Test loss: %f\r\n" % (scores))
        f.close()
        return scores


    def generate_test_images(self, input_img_array, gt_img_array):

        #TODO 3.1#NEED TO DO!!!!!

        input_img = input_img_array[0, :, :, :] * 255
        input_img = input_img.astype('uint8')
        inp_cpy = np.copy(input_img)

        gt_img = gt_img_array[0,:,:,:]*255 #(64,64,3)
        gt_img = gt_img.astype('uint8')
        full_gt = inp_cpy#uint8 type
        full_gt[32:96,32:96,:] = gt_img
        full_gt = full_gt.astype('uint8')

        pred = self.model.predict(input_img_array)
        full_pred = inp_cpy
        pred_1 = pred[0, :, :, :]*255
        pred_1 = pred_1.astype('uint8')
        full_pred[32:96,32:96,:] = pred_1
        full_pred = full_pred.astype('uint8')
        return input_img, full_gt, full_pred

    def train_adv_loss(self, nepochs):
        dataset_iterator = self.train_dataset_tf.make_one_shot_iterator()
        dataset_next_data_pair = dataset_iterator.get_next()

        for epoch in range(nepochs):            
            for _ in tqdm(range(5*self.train_dataset.num_batches)):
                input_img_array, gt_img_array = self.sess.run(
                    dataset_next_data_pair
                )

                #TODO Task 4.2
                real = np.ones((input_img_array.shape[0],1))
                fake = np.zeros((input_img_array.shape[0],1))

                self.predicted= self.generator_model.predict(input_img_array)

                disc_real = self.discriminator_model.train_on_batch(gt_img_array,real)
                disc_fake = self.discriminator_model.train_on_batch(predicted,fake)

                total_disc_loss = disc_fake + disc_real

                #GENERATOR TRAINING

                real = np.ones((input_img_array.shape[0],1))
                gen_loss = self.model.train_on_batch(input_img_array,[gt_img_array,real])

                #print("%d [D loss: %f] [G loss: %f]" % (epoch, total_disc_loss, gen_loss))

            self.show_prediction(epoch)

