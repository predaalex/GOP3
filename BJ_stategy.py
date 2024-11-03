import time

import cv2

import cv_functions
import cv2 as cv
import easyocr
import pygetwindow
import pydirectinput

dealer_card_value_position = [(770, 230), (850, 280)]
player_card_value_position = [(770, 620), (850, 670)]
game_size = [1624, 942]
hit_button_coords = [588, 875]
stand_button_coords = [811, 874]
double_button_coords = [1035, 875]
split_button_coords = [1257, 872]


def get_blackjack_action(player_total, dealer_card, is_soft=False, is_pair=False, allow_double=True, allow_surrender=False):
    """
    Determines the next action in Blackjack based on the player's hand, the dealer's upcard, and the game rules.

    Parameters:
    - player_total (int): The total value of the player's hand.
    - dealer_card (int): The dealer's visible card (2 through 11, where 11 represents an Ace).
    - is_soft (bool): Whether the hand is "soft" (contains an Ace that could be counted as 11).
    - is_pair (bool): Whether the player has a pair (two cards of the same rank).
    - allow_double (bool): Whether doubling down is allowed.
    - allow_surrender (bool): Whether surrendering is allowed.

    Returns:
    - str: The recommended action ("Hit", "Stand", "Double", "Split", or "Surrender").
    """

    # Hard hand strategy
    if not is_soft and not is_pair:
        if player_total <= 8:
            return "Hit"
        elif player_total == 9:
            if dealer_card in [3, 4, 5, 6] and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 10:
            if dealer_card in [2, 3, 4, 5, 6, 7, 8, 9] and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 11:
            if dealer_card != 11 and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 12:
            if dealer_card in [4, 5, 6]:
                return "Stand"
            return "Hit"
        elif 13 <= player_total <= 16:
            if dealer_card in [2, 3, 4, 5, 6]:
                return "Stand"
            if allow_surrender and player_total == 16 and dealer_card in [9, 10, 11]:
                return "Surrender"
            return "Hit"
        else:  # 17+
            return "Stand"

    # Soft hand strategy
    elif is_soft:
        if player_total == 13 or player_total == 14:
            if dealer_card in [5, 6] and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 15 or player_total == 16:
            if dealer_card in [4, 5, 6] and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 17:
            if dealer_card in [3, 4, 5, 6] and allow_double:
                return "Double"
            return "Hit"
        elif player_total == 18:
            if dealer_card in [3, 4, 5, 6] and allow_double:
                return "Double"
            elif dealer_card in [2, 7, 8]:
                return "Stand"
            return "Hit"
        else:  # 19+
            return "Stand"

    # Pair (split) strategy
    elif is_pair:
        if player_total == 2 or player_total == 3:
            if dealer_card in [2, 3, 4, 5, 6, 7]:
                return "Split"
            return "Hit"
        elif player_total == 4:
            if dealer_card in [5, 6]:
                return "Split"
            return "Hit"
        elif player_total == 6:
            if dealer_card in [2, 3, 4, 5, 6]:
                return "Split"
            return "Hit"
        elif player_total == 7:
            if dealer_card in [2, 3, 4, 5, 6, 7]:
                return "Split"
            return "Stand"
        elif player_total == 8:
            return "Split"
        elif player_total == 9:
            if dealer_card in [2, 3, 4, 5, 6, 8, 9]:
                return "Split"
            return "Stand"
        elif player_total == 11:  # A,A
            return "Split"
        else:  # 10,10
            return "Stand"


# Example usage
# action = get_blackjack_action(player_total=16, dealer_card=10, is_soft=False, is_pair=False)
# print(f"Recommended action: {action}")

reader = easyocr.Reader(['en'])
bj_hit_button_img = cv.imread("./resources/bj_hit_button.png", cv.IMREAD_UNCHANGED)
bj_split_button_img = cv.imread("./resources/bj_split_button.png", cv.IMREAD_UNCHANGED)
bj_double_button_img = cv.imread("./resources/bj_double_button.png", cv.IMREAD_UNCHANGED)
bj_set_your_bet_img = cv.imread("./resources/bj_set_your_bets_text.png", cv.IMREAD_UNCHANGED)
x_ads_img = cv.imread("./resources/X_ADs.png", cv.IMREAD_UNCHANGED)


def get_value(image):
    scale_factor = 3
    upscaled = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    blur = cv.blur(upscaled, (5, 5))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    processed_img = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
    text = [text[1] for text in reader.readtext(processed_img, text_threshold=0.3)]
    # print(text)
    return text[0]


def left_click(window, coords):
    x, y = coords
    pydirectinput.click(window.left + x, window.top + y)


if __name__ == '__main__':
    counter = 0
    game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]

    game_window.resizeTo(game_size[0], game_size[1])

    while True:

        game_image = cv_functions.get_image(game_window)

        player_total_image = cv_functions.extract_image(game_image, player_card_value_position)
        dealer_total_image = cv_functions.extract_image(game_image, dealer_card_value_position)

        # check if AD is display. press X for afk prevention
        if cv_functions.calculate_matchTemplate_similarity(game_image, x_ads_img) > 0.9:
            # find img coords
            top_left_x_ads_coords = cv_functions.calculate_matchTemplate_similarity(game_image, x_ads_img, get_similarity=False, get_coords=True)
            top_left_x_ads_coords = (top_left_x_ads_coords[0] + 10, top_left_x_ads_coords[1] + 10)
            left_click(game_window, top_left_x_ads_coords)
            print(f"CLOSED AD")

        # check if set your bet is required
        if cv_functions.calculate_matchTemplate_similarity(game_image, bj_set_your_bet_img) > 0.9:
            left_click(game_window, hit_button_coords)
        # check if action required
        elif cv_functions.calculate_matchTemplate_similarity(game_image, bj_hit_button_img) > 0.9:

            # get dealer and player card value
            player_total = get_value(player_total_image)
            dealer_total = get_value(dealer_total_image)

            print(f"Player Total: {player_total}")
            print(f"Dealer Total: {dealer_total}")

            if "MI" in dealer_total:
                dealer_total = "1/11"

            if player_total == "I1":
                player_total = "11"

            if player_total == "I":
                player_total = "10"

            # Errors
            if not player_total.isdigit():
                if player_total.count('/') == 1 and player_total.replace('/', '').isdigit():
                    pass
                else:
                    cv.imwrite(f"ERROR_dealer_card{counter}.png", dealer_total_image)
                    print(f"ERROR_dealer_card{counter}.png SAVED!")
                    counter += 1
                    # press stand for panik :D
                    left_click(game_window, hit_button_coords)
            if not dealer_total.isdigit():
                if dealer_total.count('/') == 1 and dealer_total.replace('/', '').isdigit():
                    pass
                else:
                    cv.imwrite(f"ERROR_player_card{counter}.png", player_total_image)
                    print(f"ERROR_player_card{counter}.png SAVED!")

                    counter += 1
                    left_click(game_window, hit_button_coords)

            # check if soft (Contains A and there is 1/11 or 2/22, etc)
            is_soft = False
            if player_total.count('/') == 1:
                is_soft = True
                player_total = player_total[-2:]

            if dealer_total.count("/") == 1:
                dealer_total = dealer_total[-2:]

            # check if split button exists
            is_pair = False
            if cv_functions.calculate_matchTemplate_similarity(game_image, bj_split_button_img) > 0.9:
                is_pair = True

            player_total = int(player_total)
            dealer_total = int(dealer_total)
            print(f"Player Total: {player_total}")
            print(f"Dealer Total: {dealer_total}")
            print(f"is_pair: {is_pair}")
            print(f"is_soft: {is_soft}")
            # decision
            decision = get_blackjack_action(int(player_total), int(dealer_total), is_soft=is_soft, is_pair=is_pair)
            print(decision)

            if decision == "Hit":
                left_click(game_window, hit_button_coords)
            elif decision == "Stand":
                left_click(game_window, stand_button_coords)
            elif decision == "Double":
                # check if double is possible (if there is a hit first, then it can't double anymore)
                if cv_functions.calculate_matchTemplate_similarity(game_image, bj_double_button_img) > 0.9:
                    left_click(game_window, double_button_coords)
                else:
                    left_click(game_window, hit_button_coords)
            elif decision == "Split":
                left_click(game_window, stand_button_coords)
            else:
                print(f"No match!")
            print("============")

        pydirectinput.moveTo(game_window.left, game_window.top)
        time.sleep(1.5)
