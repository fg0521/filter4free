import cv2
import numpy as np
import skimage
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
from models import FilmMask, Olympus, Unet
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
ttf = ImageFont.truetype('/Users/maoyufeng/shuzheng/project/algo-workshop/generator/identityCard/src/hei.ttf', 80)
# def infer2(img):
    # img = cv2.imread(img)
    # H,W,C = img.shape
    # bottom_pad_size = 0 if H%224==0 else 224-H%224
    # right_pad_size = 0 if W%224==0 else 224-W%224
    # img = np.pad(img,((0,bottom_pad_size),(0,right_pad_size), (0, 0)), 'constant', constant_values=(255, 255))
    # input = []
    # hh,ww = img.shape[0]//224,img.shape[1]//224
    # for i in range(hh): # H
    #     for j in range(ww): # W
    #         input.append(transform(Image.fromarray(img[i*224:(i+1)*224,j*224:(j+1)*224])).unsqueeze(0))
    # input = torch.from_numpy(np.array(input)).float().permute(0,3,2,1)
    # input = torch.cat(input,dim=0)
    # img = Image.open(img)
    # W,H = img.size
    # pad_w = 224*(W//224+1) if W%224!=0 else W
    # pad_h = 224*(H//224+1) if H%224!=0 else H
    # img = ImageOps.pad(img,(pad_w,pad_h),color=(0),centering=(0,0))
    # transforms.ToTensor()
    # with torch.no_grad():
    #     out = model(input)
    #
    # im = []
    # for i in range(hh):
    #     res = [cv2.cvtColor(np.array(transforms.ToPILImage()(out[i*ww+j,:,:,:].squeeze(0))),cv2.COLOR_BGR2RGB) for j in range(ww)]
    #     im.append(np.concatenate(res,axis=1))
    # im = np.concatenate(im,axis=0)
    # im = im[:H,:W]
    # cv2.imshow('test',im)
    # cv2.waitKey(0)
    # cv2.imwrite('./test10.jpg',im*255)


# 拆分图像
def split_image(image, patch_size=512):
    width, height = image.size
    patches = []
    for x in range(0, width, patch_size):
        for y in range(0, height, patch_size):
            left = x
            upper = y
            right = min(x + patch_size, width)
            lower = min(y + patch_size, height)
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
    return patches

# 合并小图像的结果
def combine_results(results, image_size, patch_size=512,org=None):
    num_cols = image_size[1] // patch_size+1
    # 创建成品图的画布
    target = Image.new('RGB', image_size)
    # 遍历每个小块的结果并将其拼接到完整图像中
    for i, result in enumerate(results):
        row = i // num_cols
        col = i % num_cols
        left = col * patch_size
        top = row * patch_size
        # right = left + patch_size
        # bottom = top + patch_size
        # print(top,left,right,bottom)
        im = Image.fromarray(torch.clamp(result.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        # ImageDraw.Draw(org).rectangle([(top,left), (top+80, left+80)], fill=None, outline="red", width=2)
        # ImageDraw.Draw(org).text((top,left), str(i), font=ttf, fill=(255, 0, 0))
        # org.show()
        # im.show()
        # print(1)
        target.paste(im, (top,left))
    return target

def infer(img,patch_size=512):
    img= Image.open(img)
    if img.size[0]*img.size[1]<=1e6:
        input = transforms.ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            out = model(input)
        target = Image.fromarray(torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    else:
        results = []
        # 对每个小块进行推理
        for patch in tqdm(split_image(img, patch_size), desc='图像推理'):
            # 对每个小块应用转换
            patch = transform(patch).unsqueeze(0)  # 添加批次维度
            # 进行推理
            output = model(patch)
            # 存储结果
            results.append(output)
        target = combine_results(results,image_size=img.size,patch_size=patch_size,org=img)
    return target

def infer4lab(img):
    img = cv2.imread(img)[:, :, ::-1]
    H,W,C = img.shape
    input = cv2.resize(img,(512,512))
    input = torch.from_numpy(skimage.color.rgb2lab(input) / 128).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        out = model(input)
    # out = out*128
    out = out.squeeze(0).permute(1, 2, 0).numpy()
    out = skimage.color.lab2rgb(out)*128
    out = cv2.resize(out,(W,H))
    # cv2.imwrite('test{epoch}.jpg',out*255)
    cv2.imshow('test',out)
    cv2.waitKey(0)


def image_concat(true_img,train_img,scaled_w=1000):
    w,h = true_img.size
    scaled_h = int(scaled_w*h/w)
    target = Image.new('RGB', (scaled_w,scaled_h*2))
    true_img = train_img.resize((scaled_w,scaled_h))
    train_img = train_img.resize((scaled_w,scaled_h))
    target.paste(true_img,(0,0))
    target.paste(train_img,(0,scaled_h))
    target.show()

if __name__ == '__main__':
    model = FilmMask()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('checkpoints/olympus/best.pth', map_location=device))
    model.to(device)
    target = infer('/Users/maoyufeng/Downloads/浓郁色调/PA044732.jpg')
    true_img = Image.open('/Users/maoyufeng/Downloads/浓郁色调/PA044732_mask.jpg')
    image_concat(true_img=true_img,train_img=target)
