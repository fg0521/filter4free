from PIL import Image
import os


def images2gif(dir,gif_name):
    # 打开图像文件并将它们添加到图像列表中
    images = []
    for img_name in sorted(os.listdir(dir)):
        if img_name.endswith('jpg'):
            image = Image.open(os.path.join(dir,img_name))
            images.append(image)

    # 将图像列表保存为 GIF 动画
    images[0].save(
        os.path.join(dir,f"{gif_name}.gif"),
        save_all=True,
        append_images=images[1:],
        duration=300,  # 每帧的持续时间（以毫秒为单位）
        loop=0 # 设置为0表示无限循环
    )

if __name__ == '__main__':
    images2gif(dir='/Users/maoyufeng/slash/project/filter-simulation/test/film_mask/mse+hist',
               gif_name='olympus')