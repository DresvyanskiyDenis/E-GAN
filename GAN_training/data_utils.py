import os.path
from typing import Optional, Callable, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf

Tensorflow_Callable = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]

def load_AffectNet_labels_original(path:str)->pd.DataFrame:
    """
    Loads the AffectNet labels from the given path.
    The resulting DataFrame has columns: [relative_path, expression_label]
    :param path: str
            path to the csv file with labels
    :return: pd.DataFrame
            DataFrame with columns [relative_path, expression_label]
    """
    df = pd.read_csv(path)
    df=df[['subDirectory_filePath','expression']]
    df.columns=['relative_path','expression_label']

    df['relative_path'] = df['relative_path'].apply(lambda x: x.replace('/', '\\'))
    df['relative_path'] = df['relative_path'].apply(lambda x: x.replace('jpg', 'png'))
    df['relative_path'] = df['relative_path'].apply(lambda x: x.replace('jpeg', 'png'))
    df['relative_path'] = df['relative_path'].apply(lambda x: x.replace('JPEG', 'png'))

    df['full_path'] = df['relative_path'].apply(lambda x: os.path.join("C:\\Users\\Professional\\Desktop\\resized",x))
    print(df.shape)
    df = df[df["full_path"].apply(os.path.isfile)]
    df = df.drop(columns=['full_path'])
    print(df.shape)
    df.to_csv("new_labels.csv", index=False)
    return


def load_AffectNet_labels_cleaned(path:str)->pd.DataFrame:
    """
    Loads the AffectNet labels from the given path.
    The resulting DataFrame has columns: [relative_path, expression_label]
    :param path: str
            path to the csv file with labels
    :return: pd.DataFrame
            DataFrame with columns [relative_path, expression_label]
    """
    df = pd.read_csv(path)
    df=df[['relative_path','expression_label']]
    df.columns=['relative_path','expression_label']
    return df

def scale_image(image:tf.Tensor):
    image = tf.cast(image, tf.float32)
    return (image - 127.5) / 127.5


def load_image(path_to_image):
    # read the image from disk, decode it, convert the data type to
    # floating point
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def get_tensorflow_image_loader(paths_to_images: pd.DataFrame, batch_size: int,
                             preprocessing_function: Optional[Tensorflow_Callable] = None,
                             clip_values: Optional[bool] = None,
                             cache_loaded_images:Optional[bool]=None) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE
    # create tf.data.Dataset from provided paths to the images and labels
    dataset = tf.data.Dataset.from_tensor_slices(paths_to_images.iloc[:, 0])
    # define shuffling
    dataset = dataset.shuffle(paths_to_images.shape[0])
    # define image loading function
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    # cache for better performance if specified
    if cache_loaded_images:
        dataset = dataset.cache()
    # create batches
    dataset = dataset.batch(batch_size)
    # apply preprocessing function to images
    if preprocessing_function:
        dataset = dataset.map(lambda x: preprocessing_function(x))
    # clip values to [0., 1.] if needed
    if clip_values:
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y))
    dataset = dataset.prefetch(AUTOTUNE)

    # done
    return dataset