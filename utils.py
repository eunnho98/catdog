import os
import random
import shutil
import piexif
from models.cnn import CNNClassifier
from torch.utils.data import random_split

def train_test_split(src_folder, train_size = 0.8):
	# Make sure we remove any existing folders and start from a clean slate
	shutil.rmtree(src_folder+'Train/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Train/Dog/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Dog/', ignore_errors=True)

	# Now, create new empty train and test folders
	os.makedirs(src_folder+'Train/Cat/')
	os.makedirs(src_folder+'Train/Dog/')
	os.makedirs(src_folder+'Test/Cat/')
	os.makedirs(src_folder+'Test/Dog/')

	# Get the number of cats and dogs images
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
	for file in files_to_be_removed:
		cat_images.remove(file)
	num_cat_images = len(cat_images)
	num_cat_images_train = int(train_size * num_cat_images)
	num_cat_images_test = num_cat_images - num_cat_images_train

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	files_to_be_removed = ['Thumbs.db', '11702.jpg']
	for file in files_to_be_removed:
		dog_images.remove(file)
	num_dog_images = len(dog_images)
	num_dog_images_train = int(train_size * num_dog_images)
	num_dog_images_test = num_dog_images - num_dog_images_train

	# Randomly assign images to train and test
	cat_train_images = random.sample(cat_images, num_cat_images_train)
	for img in cat_train_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Train/Cat/')
	cat_test_images  = [img for img in cat_images if img not in cat_train_images]
	for img in cat_test_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Test/Cat/')

	dog_train_images = random.sample(dog_images, num_dog_images_train)
	for img in dog_train_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Train/Dog/')
	dog_test_images  = [img for img in dog_images if img not in dog_train_images]
	for img in dog_test_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Test/Dog/')

	# remove corrupted exif data from the dataset
	remove_exif_data(src_folder+'Train/')
	remove_exif_data(src_folder+'Test/')

# helper function to remove corrupt exif data from Microsoft's dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	for img in cat_images:
		try:
			piexif.remove(src_folder+'Cat/'+img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	for img in dog_images:
		try:
			piexif.remove(src_folder+'Dog/'+img)
		except:
			pass

from torchvision.datasets import ImageFolder
def load_img(train_transform, test_transform=None, is_train=True):
    if is_train:
        return ImageFolder(root='../CNN/PetImages/Train', transform=train_transform)
    else:
        if test_transform is None:
            raise ValueError("test_transform is required when is_train is False")
        return ImageFolder(root='../CNN/PetImages/Test', transform=test_transform)


import torch
def split_data(train, train_ratio=.8):
	train_cnt = int(len(train) * train_ratio)
	valid_cnt = len(train) - train_cnt

	train_dataset, valid_dataset = random_split(train, [train_cnt, valid_cnt] )
	return train_dataset, valid_dataset
	
def get_hidden_sizes(input_size, output_size, n_layers):
  step_size = int((input_size - output_size) / n_layers)

  hidden_sizes = []
  current_size = input_size
  for i in range(n_layers - 1):
    hidden_sizes += [current_size - step_size]
    current_size = hidden_sizes[-1]

  return hidden_sizes


def get_model(input_size, output_size, config, channel_num, device):
	if config.model == 'cnn':
		# CNN doesn't need input size
		model = CNNClassifier(output_size=output_size, channel_num=channel_num)
	else:
		raise NotImplementedError

	return model

# src_folder = 'Dataset/PetImages/'
# train_test_split(src_folder)