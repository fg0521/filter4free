import os
import random
import time
import cv2
from tqdm import tqdm


def split_image(input_path,output_path,gap=200,image_size=512,random_clip=False,random_cnt=600):
    img_list = [i for i in os.listdir(input_path) if i.endswith('jpg') and 'mask' not in i]
    for img_name in tqdm(img_list):
        mask_img_name = img_name.replace('.jpg','')+'_mask.jpg'
        mask_im = cv2.imread(os.path.join(input_path,mask_img_name))
        org_im = cv2.imread(os.path.join(input_path, img_name))
        assert mask_im.shape == org_im.shape, 'shapes are not equal!'
        H, W, C = mask_im.shape
        if random_clip:
            # 随机裁剪
            for i in range(random_cnt):
                name = str(time.time()).replace('.', '')
                mask_name = name+'_mask.jpg'
                name = name+'.jpg'
                file = 'val' if i > int(random_cnt*0.9) else 'train'
                x, y = random.randint(0, W - 1200), random.randint(0, H - 1200)
                w, h = random.randint(800, 1200), random.randint(800, 1200)
                if x + w <= W and y + h <= H:
                    cv2.imwrite(os.path.join(output_path,file,mask_name), mask_im[y:y + h, x:x + w])
                    cv2.imwrite(os.path.join(output_path,file,name), org_im[y:y + h, x:x + w])
                else:
                    ww = W if x + w > W else x + w
                    hh = H if y + h > H else y + h
                    cv2.imwrite(os.path.join(output_path,file,mask_img_name), mask_im[y:hh, x:ww])
                    cv2.imwrite(os.path.join(output_path,file,img_name), org_im[y:hh, x:ww])
        else:
            x = 0
            cnt = 1
            while x <= W - image_size:
                x += gap
                y = 0
                while y <= H - image_size:
                    y += gap
                    name = str(time.time()).replace('.', '')
                    mask_name = name + '_mask.jpg'
                    name = name + '.jpg'
                    file = 'val' if cnt % 10 == 0 else 'train'
                    masked = mask_im[y:y + image_size, x:x + image_size]
                    if masked.shape[0]==image_size and masked.shape[1]==image_size:
                        cv2.imwrite(os.path.join(output_path,file,mask_name), mask_im[y:y + image_size, x:x + image_size])
                        cv2.imwrite(os.path.join(output_path,file,name), org_im[y:y + image_size, x:x + image_size])
                        cnt += 1

    




if __name__ == '__main__':
    split_image(input_path='/Users/maoyufeng/Downloads/浓郁色调',
                output_path='/Users/maoyufeng/Downloads/olbs2',
                random_clip=True,
                random_cnt=40)


