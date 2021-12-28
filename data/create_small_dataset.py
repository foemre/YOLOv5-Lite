import numpy as np
import json
import os
import errno
import random

def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def main():
    dirname = os.path.dirname(__file__)
    train_json_path = os.path.join(dirname, "annotations", "train.json") # potential problem: fix later
    test_json_path = os.path.join(dirname, "annotations", "test.json")
    with open(train_json_path, 'r+') as train_json, open(test_json_path, 'r+') as test_json:
        train = dict(json.load(train_json))
        test = dict(json.load(test_json))

        # Get all the images and their IDs into a dictionary
        train_files = list()
        for item in train['images']:
            image_info = [item['id'], item['file_name'], item['width'], item['height']]
            train_files.append(image_info)
        test_files = list()
        for item in test['images']:
            image_info = [item['id'], item['file_name'], item['width'], item['height']]
            test_files.append(image_info)
        # Now I have ID - filename pairs
        train_ripened = list()
        train_semi_ripened = list()
        train_green = list()
        for item in train['annotations']:
            if item['category_id'] == 3 or item['category_id'] == 6:
                train_ripened.append(item)
        for item in train['annotations']:
            if item['category_id'] == 2 or item['category_id'] == 5:
                train_semi_ripened.append(item)
        for item in train['annotations']:
            if item['category_id'] == 1 or item['category_id'] == 4:
                train_green.append(item)

        test_ripened = list()
        test_semi_ripened = list()
        test_green = list()
        for item in test['annotations']:
            if item['category_id'] == 3 or item['category_id'] == 6:
                test_ripened.append(item)
        for item in test['annotations']:
            if item['category_id'] == 2 or item['category_id'] == 5:
                test_semi_ripened.append(item)
        for item in test['annotations']:
            if item['category_id'] == 1 or item['category_id'] == 4:
                test_green.append(item)

        train_ripened = random.sample(train_ripened, 32)
        train_semi_ripened = random.sample(train_semi_ripened, 32)
        train_green = random.sample(train_green, 32)

        test_ripened = random.sample(test_ripened, 8)
        test_semi_ripened = random.sample(test_ripened, 8)
        test_green = random.sample(test_green, 8)

        train_random = train_ripened.extend(train_semi_ripened).extend(train_green)
        test_random = test_ripened.extend(test_semi_ripened).extend(test_green)

        for image_id, filename, width, height in train_files:
            for item in train_random:
                if item['image_id'] == image_id:
                    # convert COCO bounding boxes to YOLO format
                    # COCO has coords of the top left corner & width & height in pixels
                    # YOLO has coords of the center & width & height in normalized [0-1] format
                    x,y,w,h = item['bbox']
                    center_x = x + w/2
                    center_y = y + h/2
                    norm_x = center_x / width
                    norm_y = center_y / height
                    norm_w = w / width
                    norm_h = h / height
                    # merge small tomatoes and large ones
                    if item['category_id'] == 3 or item['category_id'] == 6:
                        category_id = 2
                    else:
                        category_id = item['category_id'] % 3 - 1
                    make_dir(os.path.join(dirname, "labels", "train"))
                    imgpath = os.path.join(dirname, "train", os.path.splitext(filename)[0] + ".jpg")
                    if os.path.isfile(imgpath):
                        print(filename + " train")
                        labelfile = os.path.join(dirname, "labels", "train", os.path.splitext(filename)[0] + ".txt")
                        with open(labelfile, 'a') as label:
                            line_to_write = " ".join([str(category_id), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                            label.write(line_to_write + "\n")

        for image_id, filename, width, height in test_files:
            for item in test_random:
                if item['image_id'] == image_id:
                    # convert COCO bounding boxes to YOLO format
                    # COCO has coords of the top left corner & width & height in pixels
                    # YOLO has coords of the center & width & height in normalized [0-1] format
                    x,y,w,h = item['bbox']
                    center_x = x + w/2
                    center_y = y + h/2
                    norm_x = center_x / width
                    norm_y = center_y / height
                    norm_w = w / width
                    norm_h = h / height
                    # merge small tomatoes and large ones
                    if item['category_id'] == 3 or item['category_id'] == 6:
                        category_id = 2
                    else:
                        category_id = item['category_id'] % 3 - 1
                    make_dir(os.path.join(dirname, "labels", "test"))
                    imgpath = os.path.join(dirname, "test", os.path.splitext(filename)[0] + ".jpg")
                    if os.path.isfile(imgpath):
                        print(filename + " test")
                        labelfile = os.path.join(dirname, "labels", "test", os.path.splitext(filename)[0] + ".txt")
                        with open(labelfile, 'a') as label:
                            line_to_write = " ".join([str(category_id), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                            label.write(line_to_write + "\n")
        
        for image_id, filename, width, height in train_files:
                    imgpath = os.path.join(dirname, "train", os.path.splitext(filename)[0] + ".jpg")
                    labelpath = os.path.join(dirname, "labels", "train", os.path.splitext(filename)[0] + ".txt")
                    if not os.path.isfile(labelpath):
                        os.remove(imgpath)

        for image_id, filename, width, height in test_files:
            imgpath = os.path.join(dirname, "test", os.path.splitext(filename)[0] + ".jpg")
            labelpath = os.path.join(dirname, "labels", "test", os.path.splitext(filename)[0] + ".txt")
            if not os.path.isfile(labelpath):
                print("No label")
                os.remove(imgpath)

if __name__ == "__main__":
    main()

