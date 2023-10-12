import os
import random
import time
import cv2
from tqdm import tqdm


def split_image(input_path,output_path,gap=200,image_size=512,random_clip=False,random_cnt=600):
    img_list = [i for i in os.listdir(input_path) if i.endswith('jpg') and 'org' not in i]
    for img_name in tqdm(img_list):
        org_img_name = img_name.replace('.jpg','')+'_org.jpg'
        org_im = cv2.imread(os.path.join(input_path,org_img_name))
        goal_im = cv2.imread(os.path.join(input_path, img_name))
        assert org_im.shape == goal_im.shape, 'shapes are not equal!'
        H, W, C = org_im.shape
        if random_clip:
            # 随机裁剪
            for i in range(random_cnt):
                goal_name = str(time.time()).replace('.', '')
                org_name = goal_name+'_org.jpg'
                goal_name =goal_name+'.jpg'
                file = 'val' if i > int(random_cnt*0.9) else 'train'
                x, y = random.randint(0, W - 1000), random.randint(0, H - 1000)
                w, h = random.randint(520, 1000), random.randint(520, 1000)
                if x + w <= W and y + h <= H:
                    cv2.imwrite(os.path.join(output_path,file,org_name), org_im[y:y + h, x:x + w])
                    cv2.imwrite(os.path.join(output_path,file,goal_name), goal_im[y:y + h, x:x + w])
                else:
                    ww = W if x + w > W else x + w
                    hh = H if y + h > H else y + h
                    cv2.imwrite(os.path.join(output_path,file,org_img_name), org_im[y:hh, x:ww])
                    cv2.imwrite(os.path.join(output_path,file,img_name), goal_im[y:hh, x:ww])
        else:
            x = 0
            cnt = 1
            while x <= W - image_size:
                x += gap
                y = 0
                while y <= H - image_size:
                    y += gap
                    name = str(time.time()).replace('.', '')
                    org_name = name + '_mask.jpg'
                    name = name + '.jpg'
                    file = 'val' if cnt % 10 == 0 else 'train'
                    masked = org_im[y:y + image_size, x:x + image_size]
                    if masked.shape[0]==image_size and masked.shape[1]==image_size:
                        cv2.imwrite(os.path.join(output_path,file,org_name), org_im[y:y + image_size, x:x + image_size])
                        cv2.imwrite(os.path.join(output_path,file,name), goal_im[y:y + image_size, x:x + image_size])
                        cnt += 1

    




if __name__ == '__main__':
    split_image(input_path='/Users/maoyufeng/slash/dataset/色罩',
                output_path='/Users/maoyufeng/slash/dataset/color_mask3',
                random_clip=True,
                random_cnt=100)
    # im1 = cv2.imread('/Users/maoyufeng/slash/dataset/色罩/org/9.jpg')
    # im2 = cv2.imread('/Users/maoyufeng/slash/dataset/色罩/org/9_mask.jpg')
    # H,W,_ = im1.shape
    # w = int(W/5)
    # for i in range(0,W,w):
    #     cv2.imwrite(f'/Users/maoyufeng/slash/dataset/色罩/9{i}.jpg',im1[:H,i:i+w,:])
    #     cv2.imwrite(f'/Users/maoyufeng/slash/dataset/色罩/9{i}_org.jpg',im2[:H,i:i+w,:])

