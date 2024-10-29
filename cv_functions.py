import cv2 as cv
import numpy as np
import pygetwindow
from PIL import ImageGrab
import os
import pytesseract
import simulation


left_card_coords = [(742, 560), (818, 649)]
right_card_coords = [(817, 556), (876, 649)]
flop1_coords = [(610, 357), (681, 455)]
flop2_coords = [(692, 358), (765, 454)]
flop3_coords = [(776, 358), (846, 454)]
turn_coords = [(859, 359), (929, 455)]
river_coords = [(941, 360), (1012, 454)]
pot_coords = [(672, 281), (949, 350)]
opponents_cards_coords = [
    [(470, 260), (560, 380)],
    [(1060, 260), (1160, 390)],
    [(1030, 500), (1130, 610)],
    [(490, 500), (590, 610)],
]

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
counter = 0


def convert_string_to_int(number_string):
    """
    Convert String number from pot to integer
    example: ["40000", "80000", "100000", "120000", "100000", "20.15M", "21.1M", "21.1M", "175000", "18.742999M", "19.717999M"]
    converted: [40000, 80000, 100000, 120000, 100000, 20150000, 21100000, 21100000, 175000, 18742999, 19717999]
    :param number_string:
    :return:
    """
    global counter
    try:

        # Check if the number contains 'M' (indicating millions)
        if 'M' in number_string:
            # Remove 'M' and convert to float, then multiply by 1 million
            converted_number = int(float(number_string.replace('M', '')) * 1_000_000)
        elif 'B' in number_string:
            # Remove 'B' and convert to float, then multiply by 1 billion
            converted_number = int(float(number_string.replace('B', '')) * 1_000_000_000)
        else:
            # Convert directly to integer if no 'M' is present
            converted_number = int(number_string.replace(",", ""))

        return converted_number
    except Exception as e:
        cv.imwrite(f"resources/errors{1}-{number_string}.png", game_image)
        counter += 1
        return


def get_pot_value(image):
    """
    Returns the current pot value
    """
    # get pot image
    pot_image = extract_image(image, pot_coords)

    # Apply a binary inverse threshold to make the text stand out
    _, thresh_img = cv.threshold(pot_image, 230, 255, cv.THRESH_BINARY_INV)

    # Use Gaussian blur to smooth the edges (reduce sharpness of the characters)
    blurred_img = cv.GaussianBlur(thresh_img, (3, 3), 0)

    # Apply morphological operations to make characters more distinct
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    processed_img = cv.morphologyEx(blurred_img, cv.MORPH_CLOSE, kernel)

    # OCR on processed image
    text = pytesseract.image_to_string(processed_img)

    # text to int

    pot_value = convert_string_to_int(text)

    return pot_value


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
    elif "flop" in position:
        cards_dir = "./cards_v2/flop/"

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

    if best_score > 0.80:
        return best_card
    else:
        return None


def classify_card_v2(image, position):
    """
    Returns the classified card from a certain position based on its similarity score.
    :param image:
    :param position:
    :return:
    """
    # two steps: 1. get value | 2. get rank

    # 1. get value
    value = get_card_value_or_rank(image, position, "value/")
    key = get_card_value_or_rank(image, position, "rank/")

    if key is None or value is None:
        return None

    return value, key


def get_number_of_players(image):
    """
    Returns the number of players still playing the game.
    :param image:
    :return:
    """
    players_number = 0

    for coords in opponents_cards_coords:
        possible_cards_image = extract_image(image, coords)
        for img_path in os.listdir("./player_detection_imgs"):
            template = cv.imread("./player_detection_imgs/" + img_path, cv.IMREAD_GRAYSCALE)
            similarity_score = calculate_matchTemplate_similarity(template, possible_cards_image)
            if similarity_score > 0.80:
                players_number += 1
                break
    return players_number


if __name__ == '__main__':
    game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]

    while True:

        game_image = get_image(game_window)
        cv.imshow('GOP3', game_image)
        left_card_image = extract_image(game_image, left_card_coords)
        right_card_image = extract_image(game_image, right_card_coords)
        flop1_card_image = extract_image(game_image, flop1_coords)
        flop2_card_image = extract_image(game_image, flop2_coords)
        flop3_card_image = extract_image(game_image, flop3_coords)
        turn_card_image = extract_image(game_image, turn_coords)
        river_card_image = extract_image(game_image, river_coords)
        # TODO: river card is not classified


        key = cv.waitKey(1) & 0xFF

        if key == 32:
            cv.imshow("river card", river_card_image)
            players_number = get_number_of_players(game_image)
            print(f"Number of players: {players_number}")
            print(f"Pot value: {get_pot_value(game_image)}")
            # Use an empty list if classify_card_v2 returns None
            left_hand = "".join(card for card in (classify_card_v2(left_card_image, 'left') or []) if card is not None)
            right_hand = "".join(
                card for card in (classify_card_v2(right_card_image, 'right') or []) if card is not None)

            flop1_card = "".join(
                card for card in (classify_card_v2(flop1_card_image, 'flop') or []) if card is not None)
            flop2_card = "".join(
                card for card in (classify_card_v2(flop2_card_image, 'flop') or []) if card is not None)
            flop3_card = "".join(
                card for card in (classify_card_v2(flop3_card_image, 'flop') or []) if card is not None)
            turn_card = "".join(card for card in (classify_card_v2(turn_card_image, 'flop') or []) if card is not None)
            river_card = "".join(
                card for card in (classify_card_v2(river_card_image, 'flop') or []) if card is not None)

            print(f"Player Cards: {left_hand}{right_hand}")
            print(f"Flop1 cards: {flop1_card}")
            print(f"Flop2 cards: {flop2_card}")
            print(f"Flop3 cards: {flop3_card}")
            print(f"Turn cards: {turn_card}")
            print(f"River cards: {river_card}")
            print(f"Simulation scores: {simulation.monte_carlo([left_hand, right_hand], [flop1_card, flop2_card, flop3_card, turn_card, river_card], players_number)}")
            print(f"------------------------")
