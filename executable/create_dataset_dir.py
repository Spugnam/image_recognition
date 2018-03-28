#!/usr/local/bin//python3

import os
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
            ....jpg
            ....jpg
            ....jpg
        sunflowers\
            ....jpg
        roses\
            ....jpg
"""
os.getcwd()
ROOT_DIR = ".."  # project root from file path
# destination folder
DATA_DIR = "data"
DATASET_DIR = os.path.join(ROOT_DIR, DATA_DIR, "images")

categories_file = "categories/categories.csv"
categories_file_path = os.path.join(ROOT_DIR, DATA_DIR, categories_file)

print(categories_file_path)
# source file with image urls
image_filepath = os.path.join(ROOT_DIR, DATA_DIR,
                              "raw_urls/data-with-images-000000000000.csv")


def fcount(path=DATASET_DIR):
    """
    Returns number of folders and files at path
    """
    num_dir = 0
    num_files = 0
    for root, dirs, files in os.walk(path):
        num_dir += len(dirs)
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
    parents_cat_mapping_dict =\
        pd.Series(data=categories['Parent_ID'].values,
                  index=categories['ID']).to_dict(into=dict)

    return(cat_dict, parent_cat_set, parents_cat_mapping_dict)


def is_valid_sample(row_dict, parent_cat_set, countries={'BR'},
                    exclude_=['Other', 'Estate', 'Services']):
    """
    Filter countries
    Exclude parent categories
    Exclude child categories 'Other'
    Exclude specific categories: real estate, services, tickets
    """
    if row_dict['country'] not in countries:
        return(False)
    elif row_dict['category_name'] in parent_cat_set:
        return(False)
    elif any(word in cat_dict[row_dict['category_name']] for word in exclude_):
        return(False)
    return(True)

# test
# for i, row_dict in enumerate(it.islice(f, 0, 20)):
#     print(row_dict)
#     print(cat_dict[row_dict['category_name']])
#     print(is_valid_sample(row_dict, parent_cat_set))


if __name__ == '__main__':

    # load category information
    cat_dict, parent_cat_set, parents_cat_mapping_dict =\
            load_categories(os.path.join(categories_file_path))

    # read url image file
    f = csv.DictReader(open(image_filepath, 'r'))

    # create log file name
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = now + "_image_load"
    log_filepath = os.path.join(DATASET_DIR, log_file)

    with open(log_filepath, mode='w') as log:
        for i, row_dict in enumerate(it.islice(f, 0, 2)):

            # track progress
            if i > 0 and i % 100 == 0:
                print("Saved {} images".format(i))

            # check if image should be downloaded
            if not is_valid_sample(row_dict, parent_cat_set):
                continue

            # retrieve image and update title
            try:
                im = image_from_url(row_dict['image']).resize((299, 299))
            except Exception as mess:
                log.write("{}: {}".format(mess, row_dict['title']))
                continue
            class_name = cat_dict[row_dict['category_name']]
            class_dir = os.path.join(DATASET_DIR, class_name)
            image_title = slugify(row_dict['title']) + ".jpg"

            # create class directory
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            im_path = os.path.join(class_dir, image_title)

            # log entry
            log.write("{}\t{}\t\t{}\n".format(
                i, cat_dict[row_dict['category_name']], image_title))

            # # logic to handle duplicate image
            # if not os.path.isfile(im_path):  # if no image with that title
            #     im.save(im_path)
            # else:
            #     # check if it's the same image, if not create new title
            #     # need to save image since it was changed by PIL
            #     check_im_path = os.path.join('/tmp/images_duplicates', image_title)
            #     im.save(check_im_path)
            #     if list(Image.open(im_path).getdata()) ==\
            #        list(Image.open(check_im_path).getdata()):
            #         log.write("\tduplicate image {}".format(image_title))
            #     else:
            #         im.save(get_unique_path(im_path))
            # os.remove(check_im_path)
            im.save(get_unique_path(im_path))

        print("Done. Parsed {} images".format(i))
        fcount(path=DATASET_DIR)

# EoF
