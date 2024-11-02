import cv_functions
import cv2 as cv

dealer_card_value_position = [(763, 219), (874, 280)]
player_card_value_position = [(764, 620), (871, 674)]


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
action = get_blackjack_action(player_total=16, dealer_card=10, is_soft=False, is_pair=False)
print(f"Recommended action: {action}")


if __name__ == '__main__':
    # game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]

    game_window = cv.imread("./resources/bj_image.png", cv.IMREAD_UNCHANGED)
    cv.imshow("Blackjack", game_window)

    player_total_image = cv_functions.extract_image(game_window, player_card_value_position)
    dealer_total_image = cv_functions.extract_image(game_window, dealer_card_value_position)

    # cv.imshow("player_total_image", player_total_image)
    # cv.imshow("dealer_total_image", dealer_total_image)

    text = cv_functions.image_to_text(player_total_image, 100, 255, cv.THRESH_BINARY)

    print(text)

    cv.waitKey(0)
