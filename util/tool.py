import numpy as np
import cv2


def add_text_to_image(image, text, position, font_scale=1, color=(255, 0, 0), thickness=2):
    # 在图像上添加文本
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def concatenate_images_2x2(images, texts):
    # 确保有4个图像
    if len(images) != 4:
        raise ValueError("需要4个图像来创建2x2布局")

    # 获取图像的高度和宽度
    h, w = images[0].shape[:2]

    # 创建一个空白图像，大小为2*h x 2*w
    concatenated_image = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)

    # 将图像放置在适当的位置
    positions = [(0, 0), (0, w), (h, 0), (h, w)]
    for i, (y_offset, x_offset) in enumerate(positions):
        concatenated_image[y_offset:y_offset + h, x_offset:x_offset + w] = images[i]
        add_text_to_image(concatenated_image, texts[i], (x_offset + 10, y_offset + 30))

    return concatenated_image
