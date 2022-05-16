from turtle import back
import cv2
import numpy as np
import os
import random
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
    folders = ['Bacterial_spot', 'healthy', 'Late_blight', 'Leaf_Mold', 'mosaic_virus', 'Two-spotted_spider_mite', 'Yellow_Leaf_Curl_Virus']
    images = []
    background_folder = 'complex_background'
    backgrounds = [os.path.join(background_folder, item) for item in os.listdir(os.path.join(os.getcwd(), background_folder))]
    
    dirname = os.path.dirname(__file__)
    make_dir(os.path.join(dirname, "labels", "val"))
    make_dir(os.path.join(dirname, "labels", "test"))
    make_dir(os.path.join(dirname, "labels", "train"))
    make_dir(os.path.join(dirname, "images", "val"))
    make_dir(os.path.join(dirname, "images", "test"))
    make_dir(os.path.join(dirname, "images", "train"))
    dst_val_txt = os.path.join(dirname, "labels", "val")
    dst_test_txt = os.path.join(dirname, "labels", "test")
    dst_train_txt = os.path.join(dirname, "labels", "train")
    dst_val_img = os.path.join(dirname, "images", "val")
    dst_test_img = os.path.join(dirname, "images", "test")
    dst_train_img = os.path.join(dirname, "images", "train")

    #Create list of images
    for folder in folders:
        for file in os.listdir(folder):
            images.append(os.path.join(folder, file))


    #Do 25 loops over background images
    valcount = 0
    testcount = 0
    count = 0
    while len(images) > 0:
        #For each background image
        for path in backgrounds:
            background = cv2.imread(path)
            #print(background.shape)
            background = cv2.resize(background, (1920,1200))
            
            #Limit number of images to place on background
            num_images = random.randint(5,11)
            b_rows = np.linspace(0 + random.randint(0,127), background.shape[0]-512-random.randint(0,127), num=num_images, dtype=np.uint).tolist()
            b_cols = np.linspace(0 + random.randint(0,127), background.shape[1]-512-random.randint(0,127), num=num_images, dtype=np.uint).tolist()
            
            for j in range(num_images):
                imgpath = random.choice(images)
                #print(imgpath)
                images.remove(imgpath)
                image = cv2.imread(imgpath)
                scale_ratio = random.random() + 1
                imgwidth = int(image.shape[1] * scale_ratio)
                imgheight = int(image.shape[0] * scale_ratio)
                image = cv2.resize(image,(imgwidth,imgheight))

                imgclass = os.path.split(imgpath)[0]
                imgname = os.path.split(imgpath)[1]
                rows,cols,_ = image.shape
                #b_rows = random.randint(0, background.shape[0] - 256)
                #b_cols = random.randint(0, background.shape[1] - 256)
                b_row = random.choice(b_rows)
                b_rows.remove(b_row)
                b_col = random.choice(b_cols)
                b_cols.remove(b_col)
                roi = background[b_row:b_row + rows, b_col:b_col + cols]
                
                # mask = np.zeros(image.shape[:2],np.uint8)
                # bgdModel = np.zeros((1,65),np.float64)
                # fgdModel = np.zeros((1,65),np.float64)
                # rect = (0,0,255,255)
                # cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                # image = image*mask2[:,:,np.newaxis]

                imggray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(imggray, 5, 255, cv2.THRESH_BINARY)
                imggray = cv2.GaussianBlur(imggray,(3,3),0)
                mask_inv = cv2.bitwise_not(mask)
                kernel = np.ones((3,3))
                #mask_inv = cv2.erode(mask_inv, kernel)
                mask_inv = cv2.dilate(mask_inv, kernel)
                background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img_fg = cv2.bitwise_and(image,image, mask = mask)
                dst = cv2.add(background_bg, img_fg)
                background[b_row:b_row + rows, b_col:b_col + cols] = dst
                w = image.shape[1]
                h = image.shape[0]

                #TODO:
                #Add minimum mask rect
                center_x = b_col + w/2
                center_y = b_row + h/2
                norm_x = center_x / background.shape[1]
                norm_y = center_y / background.shape[0]
                norm_w = w / background.shape[1]
                norm_h = h / background.shape[0]
                imgclass = folders.index(imgclass)

                # n_col = int(norm_x * background.shape[1] - w/2)
                # n_row = int(norm_y * background.shape[0] - h/2)


                # start_point = (int(n_col), int(n_row))
                # end_point = (int(n_col + cols), int(n_row + rows))
                # color = (0, 0, 255)
                # thickness = 3
                # background = cv2.rectangle(background, start_point, end_point, color, thickness)

                if valcount < 160:
                    with open(os.path.join(dst_val_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
                elif valcount >= 160 and testcount < 160:
                    with open(os.path.join(dst_test_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
                else:
                    with open(os.path.join(dst_train_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
            if valcount < 160:
                path_to_write = os.path.join(dst_val_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background)
                valcount += 1
            elif valcount >= 160 and testcount < 160:
                path_to_write = os.path.join(dst_test_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background)
                testcount += 1
            else:
                path_to_write = os.path.join(dst_train_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background)
            count += 1
        if len(images) == 0:
            break
if __name__ == "__main__":
    main()