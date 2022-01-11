# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

import numpy as np
import os

def main():
    base_dir = os.getcwd()
    print("base_dir: ", base_dir)
    input = base_dir + "/data/raw/"
    output = base_dir + "/data/processed/"

    train0 = np.load(input + "train_0.npz")
    train1 = np.load(input + "train_1.npz")
    train2 = np.load(input + "train_2.npz")
    train3 = np.load(input + "train_3.npz")
    train4 = np.load(input + "train_4.npz")
    
    train0_images = train0['images']
    train1_images = train1['images']
    train2_images = train2['images']
    train3_images = train3['images']
    train4_images = train4['images']

    train0_labels = train0['labels']
    train1_labels = train1['labels']
    train2_labels = train2['labels']
    train3_labels = train3['labels']
    train4_labels = train4['labels']

    test = np.load(input + "test.npz")
    
    test_images = test['images']
    test_labels = test['labels']
    
    train_images = np.concatenate((train0_images, train1_images, train2_images, train3_images, train4_images))
    train_labels = np.concatenate((train0_labels, train1_labels, train2_labels, train3_labels, train4_labels))
    
    np.savez(output + "train.npz", images = train_images, labels = train_labels)
    print(f'generated train.npz in {output}')
    np.savez(output + "test.npz", images = test_images, labels = test_labels)
    print(f'generated test.npz in {output}')

if __name__ == '__main__':
    main()