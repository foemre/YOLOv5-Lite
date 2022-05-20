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

    # for path in backgrounds:
    #     background = cv2.imread(path)
    #     background_bw = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    #     background_blur = cv2.GaussianBlur(background,(5,5),0)
    #     background_bw_blur = cv2.GaussianBlur(background_bw,(5,5),0)
    #     cv2.imwrite(os.path.splitext(path)[0]+'.jpg', background, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(path)[0]+'_bw.jpg', background_bw, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(path)[0]+'_blur.jpg', background_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(path)[0]+'_bw_blur.jpg', background_bw_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    for path in backgrounds:
        background = cv2.imread(path)
        for i, rot in enumerate([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
            background = cv2.rotate(background, rot)
            cv2.imwrite(os.path.splitext(path)[0]+'_' + str((i+1)*90) + '.jpg', background, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    print('Backgrounds done')
    for imgpath in images:
        image = cv2.imread(imgpath)
        image = cv2.GaussianBlur(image, (3,3), 0)
        cv2.imwrite(imgpath, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # for imgpath in images:
    #     image = cv2.imread(imgpath)
    #     image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     image_blur = cv2.GaussianBlur(image,(3,3),0)
    #     image_bw_blur = cv2.GaussianBlur(image_bw,(3,3),0)
    #     cv2.imwrite(os.path.splitext(imgpath)[0]+ '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(imgpath)[0]+ '_bw.jpg', image_bw, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(imgpath)[0]+ '_blur.jpg', image_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     cv2.imwrite(os.path.splitext(imgpath)[0]+'_bw_blur.jpg', image_bw_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #     for i, rot in enumerate([cv2.ROTATE_90_CLOCKWISE]):
    #         image = cv2.rotate(image, rot)
    #         image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         image_blur = cv2.GaussianBlur(image,(3,3),0)
    #         image_bw_blur = cv2.GaussianBlur(image_bw,(3,3),0)
    #         cv2.imwrite(os.path.splitext(imgpath)[0]+'_' + str((i+1)*90) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #         cv2.imwrite(os.path.splitext(imgpath)[0]+'_' + str((i+1)*90) + '_bw.jpg', image_bw, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #         cv2.imwrite(os.path.splitext(imgpath)[0]+'_' + str((i+1)*90) + '_blur.jpg', image_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #         cv2.imwrite(os.path.splitext(imgpath)[0]+'_' + str((i+1)*90) + '_bw_blur.jpg', image_bw_blur, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print('Images done')
    images = []
    for folder in folders:
        for file in os.listdir(folder):
            images.append(os.path.join(folder, file))
    
    print(len(images))

    backgrounds = [os.path.join(background_folder, item) for item in os.listdir(os.path.join(os.getcwd(), background_folder))]
    #Do 25 loops over background images
    valcount = 0
    vallimit = 160
    testcount = 0
    testlimit = 160
    count = 0
    countlimit = 999999999
    while len(images) > 0 and count < countlimit:
        #For each background image
        for path in backgrounds:
            background = cv2.imread(path)
            #print(background.shape)
            if background.shape[0] < background.shape[1]:
                background = cv2.resize(background, (1920,1080))
            else:
                background = cv2.resize(background, (1080,1920))
            
            #Limit number of images to place on background
            num_images = random.randint(4,7)
            b_rows = np.linspace(0 + random.randint(0,127), background.shape[0]-332-random.randint(1,32), num=num_images, dtype=np.uint32).tolist()
            b_cols = np.linspace(0 + random.randint(0,127), background.shape[1]-332-random.randint(1,32), num=num_images, dtype=np.uint32).tolist()
            
            for j in range(num_images):
                imgpath = random.choice(images)
                #print(imgpath)
                images.remove(imgpath)
                image = cv2.imread(imgpath)
                scale = random.random()/2 + 0.8
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)

                imgclass = os.path.split(imgpath)[0]
                imgname = os.path.split(imgpath)[1]
                #b_rows = random.randint(0, background.shape[0] - 256)
                #b_cols = random.randint(0, background.shape[1] - 256)
                
                # mask = np.zeros(image.shape[:2],np.uint8)
                # bgdModel = np.zeros((1,65),np.float64)
                # fgdModel = np.zeros((1,65),np.float64)
                # rect = (0,0,255,255)
                # cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                # image = image*mask2[:,:,np.newaxis]

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                smallkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                imggray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(imggray, 5, 255, cv2.THRESH_BINARY)
                mask = cv2.GaussianBlur(mask,(3,3),0)
                
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                ret, mask = cv2.threshold(imggray, 5, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.erode(mask, kernel, iterations=3)
                mask = cv2.erode(mask, smallkernel, iterations=3)

                contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(contours, key = cv2.contourArea)
                bndx,bndy,bndw,bndh = cv2.boundingRect(cnt)
                image = image[bndy:bndy+bndh,bndx:bndx+bndw]
                mask = mask[bndy:bndy+bndh,bndx:bndx+bndw]
                #cv2.imwrite('mask' + str(count) + '.jpg',mask)
                #cv2.imwrite('image' + str(count) + '.jpg',image)
                mask_inv = cv2.bitwise_not(mask)
                #contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # print(image.shape)
                # image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
                # image = cv2.rectangle(image, (bndx+1,bndy+1), (bndx+bndw-1,bndy+bndh-1), (0,0,255), 2)
                #get contour
                #get min rect
                #crop original image to minrect
                #proceed
                rows,cols,_ = image.shape
                b_row = random.choice(b_rows)
                b_rows.remove(b_row)
                b_col = random.choice(b_cols)
                b_cols.remove(b_col)
                filter = background 
                background = np.zeros([filter.shape[0],filter.shape[1],3],dtype=np.uint8)
                background.fill(255)
                if (b_row + rows) > background.shape[0] or (b_col + cols) > background.shape[1]:
                    print("Background shape error, auto shifting")
                    diff = int(max((b_row + rows) - background.shape[0] + 3,(b_col + cols) - background.shape[1] + 3))
                    print(diff)
                    b_row = b_row - diff
                    b_col = b_col - diff
                    roi = background[b_row:b_row + rows, b_col:b_col + cols]
                else:
                    roi = background[b_row:b_row + rows, b_col:b_col + cols]        

                background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img_fg = cv2.bitwise_and(image,image, mask = mask)
                dst = cv2.add(background_bg, img_fg)
                background[b_row:b_row + rows, b_col:b_col + cols] = dst
                background = cv2.multiply(background/255,filter/255)*255
                w = image.shape[1]
                h = image.shape[0]

                #background = cv2.rectangle(background, (b_col,b_row), (b_col + cols, b_row + rows), (0,0,255), 2)
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

                if valcount < vallimit:
                    with open(os.path.join(dst_val_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
                elif valcount >= vallimit and testcount < testlimit:
                    with open(os.path.join(dst_test_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
                else:
                    with open(os.path.join(dst_train_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
            if valcount < vallimit:
                path_to_write = os.path.join(dst_val_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                valcount += 1
            elif valcount >= vallimit and testcount < testlimit:
                path_to_write = os.path.join(dst_test_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                testcount += 1
            else:
                path_to_write = os.path.join(dst_train_img, str(count) + '.jpg')
                cv2.imwrite(path_to_write, background, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            count += 1
        if len(images) == 0:
            break
        print("Processed %d images" % count)
    print('Finished')
if __name__ == "__main__":
    main()