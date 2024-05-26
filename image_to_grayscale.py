import os

import cv2
from PIL import Image
input_dir = './simple_images/goal_post_img_1'
output_dir = './tmp/goal_post_2'
# Image.open('./simple_images/image_dataset/2631.avif')
for j in os.listdir(input_dir):
    if "avif" in j:
        continue
    else:
        image_path = os.path.join(input_dir, j).replace("\\","/")
        img_rgb = Image.open(image_path)
        img_gray = img_rgb.convert('L')
        base_filename = os.path.splitext(j)[0]
        output_filename = f'{base_filename}_gray.jpg'
        output_path = os.path.join(output_dir, output_filename)
        img_gray.save(output_path)

# image_path = './test_dataset/test16.jpg'
# img = Image.open(image_path)
# img_gray = img.convert("L")
# img_gray.show()