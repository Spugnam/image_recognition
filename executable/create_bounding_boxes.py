import os

IMAGES_DIR = '/mnt/sda1/popsy_data/raw-data/train'
BBOX_FILE_PATH = '/mnt/sda1/popsy_data/popsy_bounding_boxes.csv'


def write_bounding_boxes(path=IMAGES_DIR):
    """
    creates csv file with one line per file with the following format:
        5590670f531b3bab628b4569_filename.JPEG,0.0060,0.2620,0.7545,0.9940
    The entry can be read as:
        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

    """
    with open(BBOX_FILE_PATH, mode='w') as bbox_file:
        for root, dirs, files in os.walk(path):
            if root == path:
                continue
            if root != path:
                for file in files:
                    bbox_file.write("{},0.0,0.0,1.0,1.0\n".format(
                        # os.path.basename(root),
                        file))


if __name__ == '__main__':
    write_bounding_boxes(IMAGES_DIR)
