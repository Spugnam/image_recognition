#!/usr/bin/env python3

import os
from collections import defaultdict
import csv
from datetime import datetime
import re
import unicodedata
import itertools as it
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

"""
Create dataset with below hierarchy, to be used to create tfrecords dataset
flowers\
    flower_photos\
        tulips\
            ....JPEG
            ....JPEG
            ....JPEG
        sunflowers\
            ....JPEG
        roses\
            ....JPEG
"""

# Parameters

# number of images to parse
start = 100000
num_images = 700000
# min number of images per category (for undersampling)
min_per_cat = 1000
# image size
image_size = 299
# prevent exact duplicate images
check_duplicates = False
# use parent categories
use_parent_cat = True
# parent categories
# 559066e7531b3b96438b456c Services
# 55906545531b3baa628b4568 Electronics & Computers
# 55379c4d531b3b4c048b456b Real & Estate
# 55906559531b3b093e8b4567 Kids & Baby
# 55906530531b3b93438b456d Fashion & Accessories
# 559066d0531b3b2b478b456b Cars & Motors
# 559064a8531b3b95438b456c Art & Collectibles
# 5590654d531b3b92438b4568 Home & Garden
# 559066dd531b3b95438b456d Jobs
# 559066c7531b3bab628b4568 Tickets
# 55906718531b3b2b478b456c Animals
# 5590653b531b3b013b8b456a Sports & Outdoors
# 5590670f531b3bab628b4569 Cellphones & Tablets

# exclude categories with keywords:
exclude = ['Other', 'Estate', 'Services', 'Jobs', 'Tickets']

# directories
dirname = os.path.dirname
ROOT_DIR = dirname(dirname(os.path.abspath(__file__)))  # popsy
# destination folder
DATA_DIR = "data"
# DATASET_DIR: where to save the data
# DATASET_DIR = os.path.join(ROOT_DIR, DATA_DIR, "images/popsy_20")
DATASET_DIR = '/mnt/sda1/popsy_data/raw-data/train'

# source file with categories
categories_file = "categories/categories.csv"
categories_file_path = os.path.join(ROOT_DIR, DATA_DIR, categories_file)
# source file with image urls
image_filepath = os.path.join(ROOT_DIR, DATA_DIR,
                              "raw_urls/data-with-images-000000000000.csv")


def folder_stats(path=DATASET_DIR):
    """
    Returns number of files in each folder inside path
    """
    print("\nStatistics:", end='\n')
    num_dir = 0
    num_files = 0
    for root, dirs, files in os.walk(path):
        if root == path:
            num_dir = len(dirs)
        if root != path:
            print("Class {:60} {:>5} images".format(os.path.basename(root),
                  len(files)))
            # num_dir += len(dirs)
            num_files += len(files)
    print("number of classes: {}\
          \nnumber of images loaded: {}".format(num_dir, num_files))


def slugify(value, allow_unicode=False):
    """
    From https://github.com/django/django/blob/master/django/utils/text.py#L413
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).\
            encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def image_from_url(url):
    data = requests.get(url).content
    im = Image.open(BytesIO(data))
    return(im)


def get_unique_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.
    Example:
    get_unique_path('/etc/issue')
    '/etc/issue-1'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)
    return new_fname


def load_categories(file_path):
    """
    Returns:
    Categories dictionary
        {'55906545531b3baa628b4568': 'Electronics & Computers',
         '559068f8531b3b093e8b4568': 'Electronics & Computers - Laptops',
         '55906905531b3b93438b456e': 'Electronics & Computers - TV, Video'}
    Parent categories set
        {'55906545531b3baa628b4568', '559068f8531b3b093e8b4568', ...}
    parent_mapping_dict: Category ID -> Parent Category ID
        {'55906545531b3baa628b4568': '55906545531b3baa628b4568',
         '559068f8531b3b093e8b4568': '55906545531b3baa628b4568',
         '55906905531b3b93438b456e': '55906545531b3baa628b4568'}
    """
    # Categories
    categories = pd.read_csv(file_path)
    print("Loaded {} categories".format(categories.shape[0]))
    # Fix column names
    remove_words = ['Category', 'Name -']
    for word in remove_words:
        categories.columns = categories.columns.str.replace(word, '')
    categories.columns = categories.columns.str.strip()
    # create id to name dict
    cat_dict = pd.Series(data=categories['Name'].values,
                         index=categories['ID']).to_dict()

    # Parent categories
    parent_categories = categories.loc[pd.isna(categories['Child 1']),
                                       ['ID', 'Name', 'Parent']]
    parent_cat_set = set(parent_categories['ID'])

    # category ID to parent category mapping
    parent_IDs_dict = pd.Series(data=parent_categories['ID'].values,
                                index=parent_categories['Parent']).to_dict()
    categories['Parent_ID'] = categories['Parent'].map(parent_IDs_dict)
    parents_cat_mapping =\
        pd.Series(data=categories['Parent_ID'].values,
                  index=categories['ID']).to_dict(into=dict)
    return(cat_dict, parent_cat_set, parents_cat_mapping)


def is_valid_sample(row_dict, parent_cat_set,
                    exclude_parents=False,
                    countries={'BR'},
                    exclude=exclude):
    """
    Filter countries
    Exclude parent categories
    Exclude child categories 'Other'
    Exclude specific categories e.g. real estate, services, tickets
    """
    if row_dict['country'] not in countries:
        return(False)
    elif exclude_parents & (row_dict['category_name'] in parent_cat_set):
        return(False)
    elif any(word in cat_dict[row_dict['category_name']] for word in exclude):
        return(False)
    return(True)


if __name__ == '__main__':

    # Create dataset directory
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # load category information
    cat_dict, parent_cat_set, parents_cat_mapping =\
        load_categories(os.path.join(categories_file_path))

    # Get valid categories to track
    num_excluded_cat = {cat for cat in parent_cat_set
                        for word in exclude if word in cat_dict[cat]}
    valid_cat = parent_cat_set - num_excluded_cat
    valid_cat_counter = defaultdict(int)  # count number of images per cat
    ready_categories = len(valid_cat)  # count number of cat left to fill

    # read url image file
    f = csv.DictReader(open(image_filepath, 'r'))

    # create log file name
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = now + "_image_load"
    log_filepath = os.path.join(DATASET_DIR, log_file)

    with open(log_filepath, mode='w') as log:
        num_images_saved = 0
        for i, row_dict in enumerate(it.islice(f, start, start + num_images)):

            # track progress
            if (i > 0) and (i % (num_images // 10) == 0):
                print('>> Parsed images: {:>5} ({:.1%})\
                      \tSaved images: {}'.
                      format(i, i / num_images, num_images_saved),
                      end='\n')
                for cat, num in valid_cat_counter.items():
                    print(cat, num)

            # check if image should be downloaded
            if not is_valid_sample(row_dict, parent_cat_set):
                continue

            # get image category
            if use_parent_cat:
                # use category_code instead of description (//imagenet)
                # class_name =\
                #     cat_dict[parents_cat_mapping[row_dict['category_name']]]
                class_name =\
                    parents_cat_mapping[row_dict['category_name']]

            else:
                # use category code instead of description
                # class_name = cat_dict[row_dict['category_name']]
                class_name = row_dict['category_name']

            # check if category has already reached image maximum
            # maximum = minimum for one category * 3 (subsampling theory)
            if valid_cat_counter[class_name] >= min_per_cat * 3:
                continue

            # get additional info
            class_dir = os.path.join(DATASET_DIR, class_name)
            prefix = class_name + "_"
            image_title = prefix + slugify(row_dict['title']) + ".JPEG"

            # Download image
            try:
                # TODO: batch image download
                im = image_from_url(row_dict['image']).resize((image_size,
                                                               image_size))
            except Exception as mess:
                log.write("{}: {}".format(mess, row_dict['title']))
                continue

            # create initial target image path
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            im_path = os.path.join(class_dir, image_title)

            # logic to handle duplicate image
            if check_duplicates:
                if os.path.isfile(im_path):  # if image path already exists
                    # check if it's the same image, if not create new title
                    # need to save image since it was changed by PIL
                    check_im_path = os.path.join(DATASET_DIR, image_title)
                    im.save(check_im_path)
                    is_dup = (list(Image.open(im_path).getdata()) ==
                              list(Image.open(check_im_path).getdata()))
                    os.remove(check_im_path)  # image saved only for dup test
                    if is_dup:
                        log.write("\tduplicate image {}".format(image_title))
                        continue  # don't save the image

            # save image
            try:
                im.save(get_unique_path(im_path))
            except OSError as mess:
                try:
                    # catch OSError: cannot write mode RGBA as JPEG
                    im.convert("RGB").save(get_unique_path(im_path))
                except Exception as mess:
                    print(mess)
                    continue
            # Check if category has been filled
            valid_cat_counter[class_name] += 1
            if valid_cat_counter[class_name] == min_per_cat:
                ready_categories -= 1
                # check if all categories are filled
                if ready_categories == 0:
                    break
            num_images_saved += 1
            # log entry
            log.write("{}\t{}\t\t{}\n".format(
                num_images_saved,
                cat_dict[row_dict['category_name']],
                image_title))

        print("\nDone. Parsed {} images".format(i), end='\n')
        folder_stats(path=DATASET_DIR)
