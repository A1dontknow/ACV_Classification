import os
import numpy as np
import h5py
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from pycocotools.coco import COCO




def create_input_files(dataset, karpathy_json_path, image_folder, output_folder):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param output_folder: folder to save files
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths train, valid, set from Karpathy JSON
    train_image_paths = []
    val_image_paths = []
    test_image_paths = []


    for img in data['images']:
        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)


    # Create a base/root name for all output files
    base_filename = dataset

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)

    # impaths -> train_image_paths, imcaps -> train_image_captions, split -> TRAIN
    for impaths, split in [(train_image_paths, 'TRAIN'),
                                   (val_image_paths, 'VAL'),
                                   (test_image_paths, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            labels = h.create_dataset('labels', len(impaths), dtype='uint8')

            print("\nReading %s images, storing to file...\n" % split)

            for i, path in enumerate(tqdm(impaths)):
                # Read images
                img = load_img(impaths[i], target_size=(256, 256))
                img = img_to_array(img)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                # Save label
                annotations = coco_train.loadAnns(coco_train.getAnnIds(imgIds=int(impaths[i][-16:-4])))
                if len(annotations) == 0:
                  annotations = coco_val.loadAnns(coco_val.getAnnIds(imgIds=int(impaths[i][-16:-4])))

                # Label issue: Special image that only have back ground
                if len(annotations) == 0:
                  labels[i] = 0
                else:
                  labels[i] = annotations[0]['category_id']


if __name__ == '__main__':
    coco_train = COCO('coco/annotations/instances_train2014.json')
    coco_val = COCO('coco/annotations/instances_val2014.json')
    create_input_files(dataset='coco',
                       karpathy_json_path='coco/dataset_coco.json',
                       image_folder='coco/',
                       output_folder='coco/')