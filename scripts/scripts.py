import os
import cv2 as cv
import cv_functions

img = cv.imread("../resources/full_table.png", cv.IMREAD_UNCHANGED)

coords = [
    [(481, 267), (559, 374)],
    [(1067, 270), (1153, 380)],
    [(1040, 503), (1120, 604)],
    [(494, 501), (587, 605)],
]

for idx, coord in enumerate(coords):
    img_extracted = cv_functions.extract_image(img, coord)
    cv.imshow(f"hand{idx}", img_extracted)
    cv.imwrite(f"{idx}.png", img_extracted)
    cv.waitKey(0)
    cv.destroyAllWindows()
