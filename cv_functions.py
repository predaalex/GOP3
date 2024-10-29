import cv2 as cv
import numpy as np
import pydirectinput
import pygetwindow
from PIL import ImageGrab
import os

left_card_coords = [(742, 560), (818, 649)]  # right top | left bottom
right_card_coords = [(817, 556), (876, 649)]  # left top | right bottom
flop1 = [(610, 357), (681, 455)]
flop2 = [(692, 358), (765, 454)]
flop3 = [(776, 358), (846, 454)]
turn = [(859, 359), (929, 455)]
river = [(941, 360), (1012, 454)]


def get_image(window):
    """
    Return the image inside bbox of window as cv gray image
    """
    x, y, width, height = window.left, window.top, window.width, window.height
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

    return screenshot


def extract_image(image, coords):
    """
    Extract cropped image based on coordinates. (x1, y1) (x2, y2) Top left, Right bottom
    """
    return image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]


def calculate_matchTemplate_similarity(image, template_img):
    """
    Calculates the similarity between the template image and the image
    """
    result = cv.matchTemplate(image, template_img, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return max_val


def calculate_sift_similarity(image1, image2):
    """
    Calculates similarity between two images using SIFT feature descriptors.

    Parameters:
    - image1: First image in grayscale (OpenCV format).
    - image2: Second image in grayscale (OpenCV format).

    Returns:
    - similarity_score: A similarity score based on the number of good matches (higher is more similar).
    """
    # Initialize SIFT detector
    sift = cv.xfeatures2d.SIFT_create(nfeatures=100)

    # Detect SIFT keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Check if descriptors are found in both images
    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one or both images.")
        return 0

    # Use FLANN-based matcher for descriptor matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply the ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    # Calculate a similarity score based on the number of good matches
    similarity_score = len(good_matches)

    return similarity_score


def classify_card(image, position, algorithm="SIFT"):
    """
    Using SIFT or matchTemplate, classify the card based on its similarity score.
    """
    cards_root_path = "./cards/"
    cards_path = cards_root_path + position
    template_img_paths = os.listdir(cards_path)
    best_score = 0
    best_card = ""

    for template_img_path in template_img_paths:
        template_img = cv.imread(cards_path + template_img_path, cv.IMREAD_GRAYSCALE)

        if algorithm == "SIFT":
            similarity_score = calculate_sift_similarity(template_img, image)
        elif algorithm == "matchTemplate":
            similarity_score = calculate_matchTemplate_similarity(image, template_img)

        if similarity_score > best_score:
            best_card = template_img_path.split(".")[0]
            best_score = similarity_score
            best_img = template_img

    cv.imshow(best_card, best_img)
    cv.imshow("original", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return best_card, best_score


def get_card_value_or_rank(image, position, value_or_rank):
    """
    Classify the card rank or value based on its similarity score.
    """
    if "left" in position:
        cards_dir = "./cards_v2/left/"
    elif "right" in position:
        cards_dir = "./cards_v2/right/"
    else:  # flop # TODO: make flop as well
        return None

    cards_dir += value_or_rank
    template_img_paths = os.listdir(cards_dir)

    best_score = 0
    best_card = ""
    for template_img_path in template_img_paths:

        template_img = cv.imread(cards_dir + template_img_path, cv.IMREAD_GRAYSCALE)

        similarity_score = calculate_matchTemplate_similarity(image, template_img)

        if similarity_score > best_score:
            best_card = template_img_path.split(".")[0]
            best_score = similarity_score
            best_img = template_img

    # cv.imshow("best_match", best_img)
    # cv.imshow("original_img", image)
    # cv.waitKey(0)
    return best_card


def classify_card_v2(image, position):
    # two steps: 1. get value | 2. get rank

    # 1. get value
    value = get_card_value_or_rank(image, position, "value/")
    key = get_card_value_or_rank(image, position, "rank/")

    return value, key


if __name__ == '__main__':
    # game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]
    # game_image = get_image(game_window)
    # cv.imshow('GOP3', game_image)
    # cv.imwrite("resources/second.png", game_image)

    dir_path = './cards/right/'
    test_images = os.listdir(dir_path)

    test_images = [cv.imread(dir_path + path, cv.IMREAD_GRAYSCALE) for path in test_images]

    for image in test_images:
        # print(classify_card(image, position="right/", algorithm="matchTemplate"))
        print(classify_card_v2(image, position="right"))

