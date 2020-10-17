"""
    Converts large image dataset into train-validation-test folders with folders for each class Train-Validation-Test Split for one large image dataset 
    80-10-10 split
"""
import silence_tensorflow.auto
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import random
import shutil
import glob
import os
import re
from io import BytesIO
import requests
from zipfile import ZipFile

def get_dataset(dataset_url, data_dir, dataset_dir):
    """ retrieve weather dataset and place in data directory """
    
    print("Retrieving Dataset")
    request = requests.get(dataset_url)
    
    file = ZipFile(BytesIO(request.content))
    file.extractall(data_dir)
    ZipFile(data_dir + '/dataset2.zip').extractall(data_dir)

    os.remove(data_dir+'/dataset2.zip')
    os.rename(data_dir + '/dataset2', dataset_dir)

    print("Succesfullly Retrieved Dataset\n")

def split_data(data_dir):
    """ convert dataset into (train, val, test) with subdirectories for each image label """

    assert os.path.isdir(data_dir), f"Invalid Dataset Path: {data_dir}"

    #reading in all image file names
    files = [file for file in os.listdir(data_dir) if file.endswith((".jpg", "jpeg"))]
    random.seed(40)
    random.shuffle(files)

    split_1 = int(0.8*len(files))
    split_2 = int(0.9*len(files))

    train_files = files[:split_1]
    val_files = files[split_1:split_2]
    test_files = files[split_2:]


    #remove digits and .jpg from files
    digit_pattern = r'[0-9]'
    label_counts = Counter([(re.sub(digit_pattern,'', file)) for file in files])
    labels = list(set([label.split('.')[0] for label in label_counts.keys()]))

    train_dir =  os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    
    try:
        os.mkdir(train_dir)
        os.mkdir(val_dir)
        os.mkdir(test_dir)
    except FileExistsError:
        pass
    
    folder_path = {
        'train': train_dir, 
        'val': val_dir, 
        'test':test_dir
    }

    filenames = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split in ['train', 'val', 'test']:
        for label in labels:
            try:
                os.mkdir(os.path.join(folder_path[split],label))
            except FileExistsError:
                pass


    for split in ['train', 'val', 'test']:
        for file in filenames[split]:
            file_label = re.sub(digit_pattern, '', file).split(".")[0]
            shutil.move(data_dir + f"/{file}", folder_path[split] + f"/{file_label}") 

    print("Succesfully Split Dataset into Train, Test, Val")   

def data_augmentation(train_dir):
    """ Increasing # of training samples with random transformations """

    datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.3, 
        height_shift_range = 0.3,
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest')
    
    generator = datagen.flow_from_directory(
        train_dir,
        target_size = (150,150),
        batch_size = 32, 
        class_mode = 'categorical'
    )
    
    return generator 

if __name__ == "__main__":

    data_dir = '../data'
    dataset_dir = data_dir + '/weather_dataset'

    if(not os.path.isdir(dataset_dir)):
        dataset_url = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip'
        get_dataset(dataset_url, data_dir, dataset_dir)

    split_completed = len(glob.glob(dataset_dir+"/*.jpg")) == 0
    if(not split_completed):
        # print("hello world")
        split_data(dataset_dir)
    
    generator = data_augmentation(os.path.join(dataset_dir, 'train'))
