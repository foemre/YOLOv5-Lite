import cv2
import json
import os
import errno

'''
Create the algorithm:

1. JSON data has 6 classes of which we will be merging small ones and large ones
2. Each image has an image ID and a matching filename
3. Get all the images and their IDs into a dictionary
4. Get all bboxes of an image and its corresponding image ID into a dictionary
5. Each bbox has a corresponding class
6. Each bbox has a corresponding bbox ID
7. Open the image of an ID and crop the bboxes
8. Save the bboxes into a folder of its corresponding class with its bbox ID as filename 
9. Done

Bbox format : COCO annotation format

'''

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
        train_files = dict()
        for item in train['images']:
            train_files.update({item['id'] : item['file_name']})
        test_files = dict()
        for item in test['images']:
            test_files.update({item['id'] : item['file_name']})
        # Now I have ID - filename pairs

        # Get all bboxes, crop and save them to their respective categories
        for image_id, filename in train_files.items():
            for item in train['annotations']:
                if item['image_id'] == image_id:
                    image = cv2.imread(os.path.join(dirname, "train", filename))
                    x, y, w, h = item['bbox']
                    x = round(x)
                    y = round(y)
                    w = round(w)
                    h = round(h)
                    crop = image[y:y+h+1, x:x+w+1]
                    print(item['category_id'])
                    if item['category_id'] == 3 or item['category_id'] == 6:
                        category_id = 3
                    else:
                        category_id = item['category_id'] % 3
                    category_folder = os.path.join(dirname, "train_cropped", str(category_id))
                    make_dir(category_folder)
                    cv2.imwrite(os.path.join(category_folder, str(item['id'])) + ".jpg", crop)

        for image_id, filename in test_files.items():
            for item in test['annotations']:
                if item['image_id'] == image_id:
                    image = cv2.imread(os.path.join(dirname, "test", filename))
                    x, y, w, h = item['bbox']
                    x = round(x)
                    y = round(y)
                    w = round(w)
                    h = round(h)
                    crop = image[y:y+h+1, x:x+w+1]
                    print(item['category_id'])
                    if item['category_id'] == 3 or item['category_id'] == 6:
                        category_id = 3
                    else:
                        category_id = item['category_id'] % 3
                    category_folder = os.path.join(dirname, "test_cropped", str(category_id))
                    make_dir(category_folder)
                    cv2.imwrite(os.path.join(category_folder, str(item['id'])) + ".jpg", crop)
'''
        for image_id, filename in test_files:
            for item in train['annotations']:
                if item['image_id'] == image_id:
                    image = cv2.imread(os.path.join(dirname, "test", filename))
                    x, y, w, h = item['bbox']
                    crop = image[x:x+w+1, y:y+h+1]
                    category_id = os.path.join(dirname, str(item['category_id']))
                    make_dir(category_id)
                    cv2.imwrite(os.path.join(category_id, str(item['id'])) + ".jpg", crop)
'''
        
if __name__ == '__main__':
    main()