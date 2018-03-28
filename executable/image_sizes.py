#!/usr/local/bin//python3

import requests
from PIL import Image
from io import BytesIO


def get_image_size(url):
    # url = image_url
    data = requests.get(url).content
    im = Image.open(BytesIO(data))
    im.format
    im.mode
    im.tell()
    im.size

    out = BytesIO()
    im = Image.open(BytesIO(data))
    im.save(out, im.format)
    out.tell()  # to be verified
    return im.size


if __name__ == "__main__":
    # black heel shoes
    """
    65,564 bytes (70 KB on disk)
    """
    image_url = "https://lh3.googleusercontent.com/erfxERAP-fBPlU69XogrSNvdR-prbQvnffZleXH7G-Qmf4COq_KBKjnEa3W6cCd_GmwDqX8VeAdkwoc2FbPs=s500-e365-nu?bk=FGx4i0EUTcXjnhd0s0M4MNYeQ7MTwiwLqfG7sYv8nLjKA66YkJyqkNd0pNqPrs%2BmzI3SPE5mjqvozwUfqTxT0Sr8QE5feFU9JZH5T53OckyAe2hLZ3oU8XO1b1a%2BvNwCHdo0vLhw4kqkwehJMbVLvTS1pBhAtOdeEu5OyRf4s8KCR8qOEBhrXewNkyS742jUDBmMv7ht2puk74HgFIBcboohO2agCYJLe4tdZOHpWaCQkVVD0vEAWGDDY2nrE4HDeJSx4yR%2FIB0%2BjlZHKKv52N%2BC4fEsqAjdkHzX9tjCcqobMumVw4fmAj64ImOmR4b2nWiO3TtpLc01sH%2BVn%2FIk%2FQXDUv6pKUytzipnHQA7LTCtv%2F9R9PNPvvBK8XWA%2BXAI7T2YjoJ3SzYlnGgtDHO0nnBeGcfyVegralByU9udzx%2FD6JR8JRDwKYbnMMZMLtAmYP0IrO021p3J23VKxGv0oOcr9%2BLUAAP8pZ%2F4UT9tSNaOMU%2Bjb18LRtf%2Fy2yYvxICKgdoSvZa6ZRWa1cnFMswOA%3D%3D"  # noqa

    width, height = get_image_size(image_url)
    print("Image size: {}/{}".format(width, height))
