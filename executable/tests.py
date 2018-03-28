#!/usr/local/bin//python3

import os
import sys
import time
from datetime import datetime
import csv
import re
import itertools as it
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

image_filepath = "/Users/Quentin/Documents/Projects/popsy/data/raw_urls/data-with-images-000000000000.csv"

f = csv.DictReader(open(image_filepath, 'r'))
# odict_keys(['category_name', 'title', 'country', 'image'])

# display images
for row_dict in it.islice(f, 26, 28):
    url = row_dict['image']
    data = requests.get(url).content
    im = Image.open(BytesIO(data))
    # im.format  # JPEG
    # im.mode  # RGB
    # im.size  # (435, 500)
    print(row_dict['category_name'],
          row_dict['title'],
          row_dict['country'])
    plt.imshow(im)
    plt.axis("off")
    plt.show()
