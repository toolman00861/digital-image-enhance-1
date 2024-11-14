from util.tool import *
from util.imgEnhance import *

if __name__ == '__main__':
    configs = [
        {'img': 'img/1.bmp', 'log_param': 1, 'pow_param': 0.5},
        {'img': 'img/2.bmp', 'log_param': 1, 'pow_param': 1.1},
        {'img': 'img/3.bmp', 'log_param': 1, 'pow_param': 0.9},
        {'img': 'img/4.bmp', 'log_param': 1, 'pow_param': 1.1},
        {'img': 'img/5.tif', 'log_param': 1, 'pow_param': 0.5},
        {'img': 'img/6.tif', 'log_param': 1, 'pow_param': 0.9},
        {'img': 'img/7.tif', 'log_param': 1, 'pow_param': 1.1}
    ]
    save_path = 'out/'

    for config in configs:
        # 读取图像
        image_gray = cv2.imread(config['img'], cv2.IMREAD_GRAYSCALE)

        # 调用自己写的三种变换： 直方图均值化， log变换， 幂变换
        config['log_param'] = 255 / (np.log(1 + image_gray.max()))
        image_hist = histogram_equalization(image_gray)
        image_log = logarithmic_enhance(image_gray, config['log_param'])
        image_pow = powerlaw_enhance(image_gray, config['pow_param'])

        # 将图像转换为三通道图像，以便在上面添加文本
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        image_hist = cv2.cvtColor(image_hist, cv2.COLOR_GRAY2BGR)
        image_log = cv2.cvtColor(image_log, cv2.COLOR_GRAY2BGR)
        image_pow = cv2.cvtColor(image_pow, cv2.COLOR_GRAY2BGR)

        image_concat = concatenate_images_2x2([image_gray, image_hist, image_log, image_pow],
                                          [f'origin', f'hist',
                                           f"log_param:{.2}".format(config['log_param']),
                                           f"pow_param:{config['pow_param']}"],)

        # 储存图像
        cv2.imwrite(save_path + config['img'].split('/')[-1], image_concat)
