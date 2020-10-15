"""
    Converts large image dataset into train-validation-test folders with folders for each class Train-Validation-Test Split for one large image dataset
"""
from collections import Counter
import random
import os
import re
import shutil


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

    train_dir =  os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "testing")
    
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


if __name__ == "__main__":
    directory = "../data/weather_dataset"
    split_data(directory)
