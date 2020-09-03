import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models, regularizers, optimizers


class Model:

    def __init__(self, data_shape):
        self.data_shape = data_shape 
        self.model = None
        self.disc_model = None
        self.encoder_decoder_graph = None
        self.use_adv = False
    

    def build_model(self):
        self.create_encoder_decoder_graph()

        if not self.use_adv:
            # Change loss to reconstruction_loss_overlap when needed
            self.build_reconstruction_model(self.reconstruction_loss_overlap)
        else:
            self.build_reconstruction_model(self.reconstruction_loss_overlap)
            self.create_discriminator_graph()
            self.build_reconstruction_adversarial_model()
        #print((self.model,self.model_discriminator_graph,self.generator_model))
        return self.model #(self.model,self.model_discriminator_graph,self.generator_model)

    
    def build_reconstruction_model(self, loss):
        inputs = keras.Input(shape=self.data_shape)
        outputs = self.encoder_decoder_graph(inputs)
        print("OUTPUT of model")
        print(outputs)
        print(outputs.shape)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(lr=1.0e-3),
            loss=loss,
        )


    def build_reconstruction_adversarial_model(self):
        # Create adversarial graph
        inputs_discriminator_graph = keras.Input(shape=(64,64,3))
        output = self.discriminator_graph(inputs_discriminator_graph)

        self.model_discriminator_graph = keras.Model(
            inputs_discriminator_graph, output
        )

        self.model_discriminator_graph.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1.0e-4),
            loss_weights=[0.001]
        )#real

        # Input to the whole model
        inputs = keras.Input(shape=self.data_shape)

        # Reconstruction
        reconstruction_output = self.encoder_decoder_graph(inputs) #output of generator network
        adversarial_output = self.discriminator_graph(reconstruction_output) #taking predicted image(64x64x3) and then going through discriminator: real or fake?

        self.model = keras.Model(
            inputs, 
            outputs=[reconstruction_output, adversarial_output]
        )

        self.model.compile(
            optimizer=optimizers.Adam(lr=1.0e-4),
            loss=[self.reconstruction_loss_overlap, 'binary_crossentropy'],
            loss_weights=[0.999, 0.001]
        )

    
    def create_encoder_decoder_graph(self):
        """ 
        Creates a graph and store it in self.encoder_decoder_graph 
        No return
        """
        self.encoder_decoder_graph = models.Sequential()
        self.encoder_decoder_graph.add(layers.Conv2D(64, kernel_size=4, strides=2, input_shape=(128,128,3), padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))
        self.encoder_decoder_graph.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))
        self.encoder_decoder_graph.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))
        self.encoder_decoder_graph.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))
        self.encoder_decoder_graph.add(layers.Conv2D(512, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))
        self.encoder_decoder_graph.add(layers.Conv2D(4000, kernel_size=4, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.LeakyReLU(alpha=0.2))

        # DECODER
        self.encoder_decoder_graph.add(layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.Activation('relu'))
        self.encoder_decoder_graph.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.Activation('relu'))
        self.encoder_decoder_graph.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.Activation('relu'))
        self.encoder_decoder_graph.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.Activation('relu'))
        self.encoder_decoder_graph.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
        self.encoder_decoder_graph.add(layers.BatchNormalization())
        self.encoder_decoder_graph.add(layers.Activation('tanh'))

        #TODO Task 2.1
        pass


    def create_discriminator_graph(self):
        """
        Creates a discriminator graph in self.discriminator_graph
        """
        #TODO Task 4.1
        self.discriminator_graph = models.Sequential()
        self.discriminator_graph.add(layers.Conv2D(64, kernel_size=4, strides=2, input_shape=(64, 64, 3), padding='same',kernel_regularizer=regularizers.l2(0.01)))
        self.discriminator_graph.add(layers.BatchNormalization())
        self.discriminator_graph.add(layers.LeakyReLU(alpha=0.2))
        self.discriminator_graph.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.discriminator_graph.add(layers.BatchNormalization())
        self.discriminator_graph.add(layers.LeakyReLU(alpha=0.2))
        self.discriminator_graph.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.discriminator_graph.add(layers.BatchNormalization())
        self.discriminator_graph.add(layers.LeakyReLU(alpha=0.2))
        self.discriminator_graph.add(layers.Conv2D(512, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))
        self.discriminator_graph.add(layers.BatchNormalization())
        self.discriminator_graph.add(layers.LeakyReLU(alpha=0.2))
        self.discriminator_graph.add(layers.Flatten())
        self.discriminator_graph.add(layers.Dense(1, activation='sigmoid'))
        pass


    def reconstruction_loss(self, y_true, y_pred):
        """ 
        Creates the reconstruction loss between y_true and y_pred
        """
        #TODO Task 2.2

        mask_center = tf.pad(tf.ones([50, 50]), [[7, 7], [7, 7]])
        mask_center = tf.reshape(mask_center, [64, 64, 1])
        mask_center = tf.concat([mask_center] * 3, 2)  # 3 channels
        mask_overlap = 1 - mask_center
        loss = tf.square(y_true - y_pred)
        loss_center = tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss * mask_center, [1, 2, 3])))  # Loss for center part(50x50)
        loss_overlap = tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss * mask_overlap, [1, 2,3])))# Loss for overlapping region(the extra 7 pixels per side)
        reconstruction_loss_value = loss_center + loss_overlap
        return reconstruction_loss_value

    
    def reconstruction_loss_overlap(self, y_true, y_pred):
        """ 
        Similar to reconstruction loss, but with predicted region overlapping
        with the input
        """
        #TODO Task 2.3
        mask_center = tf.pad(tf.ones([50,50]), [[7,7], [7,7]])
        mask_center= tf.reshape(mask_center, [64, 64, 1])
        mask_center= tf.concat([mask_center] * 3,2) #3 channels
        mask_overlap = 1 - mask_center
        loss= tf.square(y_true - y_pred)
        loss_center = tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss* mask_center ,[1, 2, 3])))/10  # Loss for center part(50x50)
        loss_overlap = tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss* mask_overlap, [1, 2, 3])))  # Loss for overlapping region(the extra 7 pixels per side)
        reconstruction_loss_value = loss_center + loss_overlap
        return reconstruction_loss_value





