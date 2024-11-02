import cv_functions
import cv2 as cv
import time
import pygetwindow
import pydirectinput

button_coords = [(691, 710), (935, 809)]
click_button_coords = [(691 + 935) // 2, (710 + 809) // 2]
click_spin_coords = [1333, 861]

collect_img = cv_functions.extract_image(cv.imread(r"resources/slot_machine_collect.png", cv.IMREAD_GRAYSCALE), button_coords)
continue_img = cv_functions.extract_image(cv.imread(r"resources/slot_machine_continue.png", cv.IMREAD_GRAYSCALE), button_coords)
start_img = cv_functions.extract_image(cv.imread(r"resources/slot_machine_start.png", cv.IMREAD_GRAYSCALE), button_coords)

game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]


def left_click(window, coords):
    x, y = coords
    pydirectinput.click(window.left + x, window.top + y)


def check_buttons(game_img, threshold=0.9):
    cv.imshow("check_buttons", game_img)

    # check if collect/start/continue button exists
    collect_similarity = cv_functions.calculate_matchTemplate_similarity(game_img, collect_img)
    continue_similarity = cv_functions.calculate_matchTemplate_similarity(continue_img, continue_img)
    start_similarity = cv_functions.calculate_matchTemplate_similarity(start_img, start_img)
    similarity = max(collect_similarity, continue_similarity, start_similarity)

    if similarity > threshold:
        left_click(game_window, click_button_coords)


if __name__ == '__main__':
    counter = 0


    while True:
        curr_game_img = cv_functions.get_image(game_window)

        # press spin button
        left_click(game_window, click_spin_coords)
        counter += 1

        # check if there is collect/continue/start every 10 clicks
        if counter % 10 == 0:
            check_buttons(curr_game_img)
            counter = 0

        time.sleep(0.2)
