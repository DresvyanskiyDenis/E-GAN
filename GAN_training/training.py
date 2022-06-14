import keras.callbacks
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt

from GAN_training.data_utils import load_AffectNet_labels_cleaned, get_tensorflow_image_loader, scale_image
from GAN_training.utils import create_generator_only_faces, create_discriminator_only_faces, GAN

class generator_checker_callback(tf.keras.callbacks.Callback):
    def __init__(self, latent_vector, generator):
        self.latent_vector=latent_vector
        self.generator=generator


    def check_generator_quality(self, latent_vectors:np.ndarray, generator:tf.keras.Model, epoch:int):
        generated_images = generator.predict(latent_vectors)
        num_rows, num_cols= generated_images.shape[0]//4, generated_images.shape[0]//generated_images.shape[0]//4
        f, axarr = plt.subplots(num_rows, num_cols)
        for i in range(generated_images.shape[0]):
            axarr[i]=generated_images[i]
        plt.savefig("generated_images_epoch_%i.png"%epoch)

    def on_epoch_end(self, epoch, logs={}):
        self.check_generator_quality(self.latent_vector, self.generator, epoch)


def main():
    # TODO: create the genertor quality checking (image generation)
    # paths
    path_to_labels="C:\\Users\\Professional\\Desktop\\AffectNet_Labels.csv"
    path_to_data="C:\\Users\\Professional\\Desktop\\resized"
    # params for training
    latent_dim = 128
    generated_normal_vector=np.random.standard_normal(size=(16,latent_dim))
    input_generator=(latent_dim,)
    input_discriminator = (224,224,3)
    batch_size = 64
    learning_rate=0.0001
    # load and prepare labels
    labels=load_AffectNet_labels_cleaned(path_to_labels)
    print(labels.shape)
    print(labels)
    labels['relative_path']=labels['relative_path'].apply(lambda x: os.path.join(path_to_data,x))
    labels['expression_label']=1
    train_generator=get_tensorflow_image_loader(paths_to_images= labels, batch_size= batch_size,
                             preprocessing_function = scale_image,
                             clip_values = None,
                             cache_loaded_images=None)

    #for images in train_generator.as_numpy_iterator():
    #    print(images.shape)

    # GAN model initialization
    generator=create_generator_only_faces(input_generator)
    discriminator=create_discriminator_only_faces(input_discriminator)
    checker_callback = generator_checker_callback(generated_normal_vector, generator)
    gan=GAN(discriminator, generator, latent_dim)
    gan.compile(d_optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                g_optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=False))
    gan.fit(train_generator, epochs=100, callbacks=[checker_callback])


if __name__ == '__main__':
    main()