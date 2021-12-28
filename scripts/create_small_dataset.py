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

def tomato_process(coco_json_path=None):
    with open(coco_json_path, 'r+') as json_contents:
        contents = json.load(json_contents)

        files = list()
        for item in contents['images']:
            image_info = [item['id'], item['file_name'], item['width'], item['height']]
            files.append(image_info)
        
        ripened = list()
        semi_ripened = list()
        green = list()
        for item in contents['annotations']:
            if item['category_id'] == 3 or item['category_id'] == 6:
                ripened.append(item)
        for item in contents['annotations']:
            if item['category_id'] == 2 or item['category_id'] == 5:
                semi_ripened.append(item)
        for item in contents['annotations']:
            if item['category_id'] == 1 or item['category_id'] == 4:
                green.append(item)

        ripened = random.sample(ripened, 32)
        semi_ripened = random.sample(semi_ripened, 32)
        green = random.sample(green, 32)
        randomized = ripened + semi_ripened + green

        folder_name = os.path.basename(coco_json_path).split('.')[0]

        for image_id, filename, width, height in files:
            for item in randomized:
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
                    make_dir(os.path.join(dirname, "labels", folder_name))
                    imgpath = os.path.join(dirname, folder_name, os.path.splitext(filename)[0] + ".jpg")
                    if os.path.isfile(imgpath):
                        labelfile = os.path.join(dirname, "labels", folder_name, os.path.splitext(filename)[0] + ".txt")
                        with open(labelfile, 'a') as label:
                            line_to_write = " ".join([str(category_id), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                            label.write(line_to_write + "\n")

        for image_id, filename, width, height in files:
            imgpath = os.path.join(dirname, folder_name, os.path.splitext(filename)[0] + ".jpg")
            labelpath = os.path.join(dirname, "labels", folder_name, os.path.splitext(filename)[0] + ".txt")
            if not os.path.isfile(labelpath):
                try:
                    os.remove(imgpath)
                except OSError:
                    pass

def main():
    train_json_path = os.path.join(dirname, "annotations", "train.json") # potential problem: fix later
    test_json_path = os.path.join(dirname, "annotations", "test.json")
    tomato_process(train_json_path)
    tomato_process(test_json_path)

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    main()

