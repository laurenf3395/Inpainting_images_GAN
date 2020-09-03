from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import skimage
import tensorflow as tf
from tensorflow import keras 

from model import Model
from trainer import Trainer 
from dataset import Dataset
import tensorflow.contrib.eager as tfe


def main():
    train_data_path = '/cluster/scratch/laurenf/HW6/Handout/dataset/train'
    val_data_path = '/cluster/scratch/laurenf/HW6/Handout/dataset/val'
    test_data_path = '/cluster/scratch/laurenf/HW6/Handout/dataset/test'

    # Read the data
    train_dataset = Dataset(train_data_path)
    val_dataset   = Dataset(val_data_path)
    test_dataset  = Dataset(test_data_path)

    # Create dataset
    batch_size = 40

    train_dataset.create_tf_dataset(batch_size) #from dataset.py : into tensor
    val_dataset.create_tf_dataset(batch_size)
    test_dataset.create_tf_dataset(batch_size)

    # Create the model
    model = Model([128,128,3])

    # Train the model
    trainer = Trainer(
        model, train_dataset, val_dataset, test_dataset
    )
    trainer.train(nepochs=2000)
    trainer.test(20)


if __name__ == '__main__':
    main()
    