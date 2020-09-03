import glob
import numpy as np 
import os
import skimage
import tensorflow as tf 
from PIL import Image
from tensorflow.python.keras import backend as K

use_random_crop = True


class Dataset:

    def __init__(self, data_path):
        self.data_path = data_path
        self.read_data(data_path)

        self.num_data = self.data.shape[0]
        self.batch_size = None
        self.num_batches = None

        if not use_random_crop:
            self.crop_center_img()


    def read_data(self, data_path):
        # Get list of files
        data = []

        for img_type in ["jpg", "png"]:
            for img_path in glob.glob("{}/*.{}".format(data_path, img_type)):
                img_name = os.path.basename(img_path)
                
                img = skimage.io.imread(img_path).astype(np.float32)
                img/= 255.0

                # Append to output list
                data.append(img.astype(np.float32))

        self.data = np.stack(data, axis=0)

        self.mean = self.data.mean()
        self.std  = self.data.std()


    def create_tf_dataset(self, batch_size=32):
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.num_data / batch_size).astype(int)

        self.dataset_tf = tf.data.Dataset.from_tensor_slices(self.data)
        #print(self.dataset_tf.shape)
        if use_random_crop:
            self.dataset_tf = self.dataset_tf.map(self.crop_random_img)

        self.dataset_tf = self.dataset_tf.shuffle(200)
        self.dataset_tf = self.dataset_tf.batch(batch_size)
        self.dataset_tf = self.dataset_tf.repeat()            


    def crop_center_img(self):
        # TODO Task 1.1
        """
        Modifies self.data in order to remove the center region
        self.data should now be a tuple 
        (img_with_missing_crop, groundtruth_crop)
        """
        img = self.data
        img_with_missing_crop = np.copy(img)
        dim =128
        crop = dim // 2
        start = crop - (crop // 2)
        #ground truth overlaps img_with_missing_crop by 7 pixels in all directions
        img_with_missing_crop[:,start+7:start + crop-7, start+7:start + crop-7,:] = 0
        #255
        #inpu = Image.fromarray((img_with_missing_crop[1,:,:,:]*255).astype('uint8'))
        #inpu.save("cropped.png")
        groundtruth_crop = img[:,start:start + crop, start:start + crop,:]
        self.data = (img_with_missing_crop, groundtruth_crop)




    def crop_random_img(self,input_img):
        #TODO Task 1.2
        """
        Modifies the input_img in order to remove a random region
        It must return the following tuple 
        (img_with_missing_crop, groundtruth_crop)
        """

        img_with_missing = tf.identity(input_img)
        #print("IMG_MISSING_CROP SIZE")
        #print(img_with_missing.shape)
        dim = 128
        size_box = 64
        random_y = np.random.randint(0,63)#(7, 70)
        random_x = np.random.randint(0, 63)#size_box - 7)
        # ground truth overlaps img_with_missing_crop by 7 pixels in all directions
        mask = np.ones([128, 128, 3], np.float32)
        mask_black = mask[random_y + 7:random_y + size_box -7 , random_x+7:random_x + size_box-7, :]
        mask[random_y + 7:random_y + size_box -7 , random_x+7:random_x + size_box-7, :] = 0
        mask_tensor = tf.constant(mask, tf.float32)
        img_with_missing_crop = np.multiply(img_with_missing,mask_tensor)

        mask_gt = np.zeros([128, 128, 3], np.float32)
        mask_gt[random_y :random_y + size_box, random_x:random_x + size_box, :] = 1
        mask_gt_tensor = tf.constant(mask_gt, tf.float32)
        groundtruth= np.multiply(img_with_missing, mask_gt_tensor)
        groundtruth_crop = tf.image.crop_to_bounding_box(groundtruth,random_y,random_x,size_box,size_box)


        input_img = (img_with_missing_crop,groundtruth_crop)
        return input_img
        #pass