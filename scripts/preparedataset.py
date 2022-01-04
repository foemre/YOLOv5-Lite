import cv2
import numpy as np
import json
import os
import errno
import shutil

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
    val_files = [
        'IMG_0989', 'IMG_0992', 'IMG_0994', 'IMG_0997', 
        'IMG_0999', 'IMG_1003', 'IMG_1012', 'IMG_1016', 
        'IMG_1022', 'IMG_1038', 'IMG_1051', 'IMG_1061', 
        'IMG_1065', 'IMG_1066', 'IMG_1068', 'IMG_1085', 
        'IMG_1092', 'IMG_1095', 'IMG_1096', 'IMG_1104', 
        'IMG_1105', 'IMG_1113', 'IMG_1114', 'IMG_1123', 
        'IMG_1131', 'IMG_1132', 'IMG_1136', 'IMG_1142', 
        'IMG_1152', 'IMG_1169', 'IMG_1181', 'IMG_1190', 
        'IMG_1194', 'IMG_1214', 'IMG_1224', 'IMG_1236', 
        'IMG_1253', 'IMG_1254', 'IMG_1265', 'IMG_1271', 
        'IMG_1286', 'IMG_1293', 'IMG_1298', 'IMG_1299', 
        'IMG_20191215_110454', 'IMG_20191215_110636', 'IMG_20191215_110647', 'IMG_20191215_110658', 
        'IMG_20191215_110717', 'IMG_20191215_110730', 'IMG_20191215_110752', 'IMG_20191215_110813', 
        'IMG_20191215_110842', 'IMG_20191215_110851', 'IMG_20191215_110902', 'IMG_20191215_110913', 
        'IMG_20191215_111007', 'IMG_20191215_111013', 'IMG_20191215_111044', 'IMG_20191215_111121', 
        'IMG_20191215_111144', 'IMG_20191215_111228', 'IMG_20191215_111236', 'IMG_20191215_111237', 
        'IMG_20191215_111256', 'IMG_20191215_111316', 'IMG_20191215_111338', 'IMG_20191215_111341', 
        'IMG_20191215_111343', 'IMG_20191215_111356', 'IMG_20191215_111357_1', 'IMG_20191215_111417', 
        'IMG_20191215_111427', 'IMG_20191215_111444', 'IMG_20191215_111501', 'IMG_20191215_111512', 
        'IMG_20191215_111532', 'IMG_20191215_111548', 'IMG_20191215_111552', 'IMG_20191215_111631', 
        'IMG_20191215_111638', 'IMG_20191215_111654', 'IMG_20191215_111751', 'IMG_20191215_111857', 
        'IMG_20191215_111905', 'IMG_20191215_111933', 'IMG_20191215_111947', 'IMG_20191215_112012_1', 
        'IMG_20191215_112024', 'IMG_20191215_112307', 'IMG_20191215_112325', 'IMG_20191215_112356', 
        'IMG_20191215_112531', 'IMG_20191215_112539', 'IMG_20191215_112619', 'IMG_20191215_112715', 
        'IMG_20191215_112718', 'IMG_20191215_112807', 'IMG_20191215_112843', 'IMG_20191215_112854']
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

        for image_id, filename, width, height in train_files:
            for item in train['annotations']:
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
                    labelfile = os.path.join(dirname, "labels", "train", os.path.splitext(filename)[0] + ".txt")
                    with open(labelfile, 'a') as label:
                        line_to_write = " ".join([str(category_id), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        label.write(line_to_write + "\n")

        for image_id, filename, width, height in test_files:
            for item in test['annotations']:
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
                    labelfile = os.path.join(dirname, "labels", "test", os.path.splitext(filename)[0] + ".txt")
                    with open(labelfile, 'a') as label:
                        line_to_write = " ".join([str(category_id), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        label.write(line_to_write + "\n")

        #move train and test files according to yolo directory structure
        make_dir(os.path.join(dirname, "labels", "val"))
        make_dir(os.path.join(dirname, "images", "val"))
        make_dir(os.path.join(dirname, "images", "test"))
        make_dir(os.path.join(dirname, "images", "train"))
        print("Dirs created")
        source_train_img = os.path.join(dirname, "train")
        destination_train_img = os.path.join(dirname, "images", "train")
        source_test_img = os.path.join(dirname, "test")
        destination_test_img = os.path.join(dirname, "images", "test")
        files_list = os.listdir(source_train_img)
        for item in files_list:
            shutil.copy(os.path.join(source_train_img, item), destination_train_img)
        print("Train imgs copied")
        files_list = os.listdir(source_test_img)
        for item in files_list:
            shutil.copy(os.path.join(source_test_img, item), destination_test_img)
        print("Test imgs copied")

        source_train_labels = os.path.join(dirname, "labels", "train")
        source_test_labels = os.path.join(dirname, "labels", "test")
        destination_val_img = os.path.join(dirname, "images", "val")
        destination_val_labels = os.path.join(dirname, "labels", "val")
        #create validation dataset
        files_list = os.listdir(destination_train_img)
        for item in val_files:
            if os.path.exists(os.path.join(destination_train_img, item + ".jpg")):
                shutil.copy(os.path.join(destination_train_img, item + ".jpg"), destination_val_img)
        print("Val imgtrain copied")
        files_list = os.listdir(destination_test_img)
        for item in val_files:
            if os.path.exists(os.path.join(destination_test_img, item + ".jpg")):
                shutil.copy(os.path.join(destination_test_img, item + ".jpg"), destination_val_img)
        print("Val imgtest copied")
        files_list = os.listdir(source_train_labels)
        for item in val_files:
            if os.path.exists(os.path.join(source_train_labels, item + ".txt")):
                shutil.copy(os.path.join(source_train_labels, item + ".txt"), destination_val_labels)
        print("Val trainlabels copied")
        files_list = os.listdir(source_test_labels)
        for item in val_files:
            if os.path.exists(os.path.join(source_test_labels, item + ".txt")):
                shutil.copy(os.path.join(source_test_labels, item + ".txt"), destination_val_labels)
        print("Val testlabels copied")
            

if __name__ == "__main__":
    main()

