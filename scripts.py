import os
import cv2
import cv_functions

# used to extract rank and value of cards img from a specific folder

imgs_dir = "./cards/tmp"
imgs_name = os.listdir(imgs_dir)

for img_name in imgs_name:
    image = cv2.imread(f"{imgs_dir}/{img_name}")
    print(f"{imgs_dir}/{img_name}")

    extracted_values_img = cv_functions.extract_image(image, [(18, 49), (59, 91)])
    file_name = f"./cards_v2/right/rank/{img_name[1]}.png"

    cv2.imwrite(file_name, extracted_values_img)
