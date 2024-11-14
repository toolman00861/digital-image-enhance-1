from util import *

if __name__ == '__main__':
    image_list = [
        'img/1.bmp', 'img/2.bmp', 'img/3.bmp', 'img/4.bmp',
        'img/5.tif', 'img/6.tif', 'img/7.tif',
    ]
    save_path = 'out/'

    for image_path in image_list:
        # 读取图像
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 调用自己写的三种变换： 直方图均值化， log变换， 幂变换
        image_hist = histogram_equalization(image_gray)
        image_log = logarithmic_enhance(image_gray, 255 / (np.log(1 + image_gray.max())))
        image_pow = powerlaw_enhance(image_gray, 0.5)

        # 储存图像
        cv2.imwrite(save_path + image_path.split('/')[-1] + '_hist.jpg', image_hist)
        cv2.imwrite(save_path + image_path.split('/')[-1] + '_log.jpg', image_log)
        cv2.imwrite(save_path + image_path.split('/')[-1] + '_power.jpg', image_pow)
