import os
from PIL import Image

def resize_image(image_path, save_path):
    image = Image.open(image_path)
    after_img = image.resize((227,227))
    after_img.save(save_path)

IMG_TYPE = ['GADF', 'MTF']
UPPER_GYM_WORKOUT = ['Dumbbell_Curl', 'DUmbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl']

for i_type in IMG_TYPE:
    base_path = './' + i_type + '/'
    save_path = './Images/' + i_type + '/'
    for folder in UPPER_GYM_WORKOUT:
        for top, dir, files in os.walk(base_path+folder):
            for file in files:
                resize_image(base_path + folder + '/' + file, save_path + folder + '/' + file)
        print(folder + ', Done')