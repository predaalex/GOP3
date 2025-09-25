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
call_button_coords = [(766, 847), (1100, 915)]
my_money_coords = [(919, 727), (1030, 765)]
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
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


def image_to_text(image, min_val=230, max_val=250, thresh_function=cv.THRESH_BINARY_INV):
    # Apply a binary inverse threshold to make the text stand out
    _, thresh_img = cv.threshold(image, min_val, max_val, thresh_function)
    # Use Gaussian blur to smooth the edges (reduce sharpness of the characters)
    blurred_img = cv.GaussianBlur(thresh_img, (3, 3), 0)
    # Apply morphological operations to make characters more distinct
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    processed_img = cv.morphologyEx(blurred_img, cv.MORPH_CLOSE, kernel)
    # OCR on processed image
    text = pytesseract.image_to_string(processed_img)
    return text


def get_pot_value(image):
    """
    Returns the current pot value
    """
    # get pot image
    pot_image = extract_image(image, pot_coords)
    cv.imshow("pot_image", pot_image)
    text = image_to_text(pot_image, thresh_function=cv.THRESH_BINARY_INV)

    # text to int
    pot_value = convert_string_to_int(text)

    return pot_value


def get_my_money(image):
    my_money_img = extract_image(image, my_money_coords)
    my_money_img = cv.resize(my_money_img, dsize=(0, 0), fx=3, fy=3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Sharpen the image
    my_money_img = cv.filter2D(my_money_img, -1, kernel)

    # black outside coords
    h, w = my_money_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # 1 channel mask, all zeros
    mask[15:87, 15:] = 255  # region you want to keep

    # If original image is grayscale:
    my_money_img = cv.bitwise_and(my_money_img, my_money_img, mask=mask)
    cv.imshow("my_money", my_money_img)
    text = image_to_text(my_money_img, thresh_function=cv.THRESH_BINARY_INV)
    my_money = convert_string_to_int(text)
    return my_money


def get_call_value(image):
    """
    Returns the current call value
    :return: Call Value(Int) | CHECK | ALL IN
    """
    # get call image
    call_image = extract_image(image, call_button_coords)
    text = image_to_text(call_image, thresh_function=cv.THRESH_BINARY_INV)

    # print(text)  # debug call value
    if "CALL ANY" in text:
        return 0
    elif "CHECK" in text:
        return 0
    elif "ALL IN" in text:
        return get_my_money(image)
    else:
        try:
            return convert_string_to_int(text.split(" ")[1])
        except Exception as e:
            print(f"could not convert call value")
            return 0


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


def calculate_matchTemplate_similarity(image, template_img, get_similarity=True, get_coords=False):
    """
    Calculates the similarity between the template image and the image
    """
    result = cv.matchTemplate(image, template_img, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    if get_similarity:
        return max_val
    if get_coords:
        return max_loc  # min_loc left top corner

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

    if best_score > 0.5:
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
    rank = get_card_value_or_rank(image, position, "rank/")

    if rank is None or value is None:
        return None

    return value, rank


def get_number_of_opponents(image):
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
            if similarity_score > 0.60:
                players_number += 1
                break
    return players_number


def compute_ev(pot_value, call_value, win_prob, lose_prob):
    """
    Your original EV-of-calling formula (lose_prob may include ties).
    Interprets:
      pot_value = pot before your action
      call_value = amount you must invest to call
    """
    if None in (pot_value, call_value, win_prob, lose_prob):
        return None
    pot_if_call = pot_value + call_value
    return (win_prob * pot_if_call) - (lose_prob * call_value)


import math
from functools import lru_cache

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def binom_pmf(n, k, p):
    # Simple binomial PMF
    if k < 0 or k > n: return 0.0
    if p <= 0: return 1.0 if k == 0 else 0.0
    if p >= 1: return 1.0 if k == n else 0.0
    # nCk
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def logistic_continue_prob(R):
    # Less fold-happy: at R=0, ~45% continue; at R=5000, ~27% continue
    a, b = -0.2, -0.0002
    return 1.0 / (1.0 + math.exp(-(a + b*float(R))))

def decide_action_pro(
    # State
    hand, board, opponents, pot_value, call_value,
    # Betting constraints
    min_raise=0, max_raise=0,
    hero_stack=None, opp_stack=None,  # effective stacks in front of you now
    # Equity model
    get_equity=None,    # function k -> (win_prob, tie_prob); if None, uses simulation.monte_carlo
    equity_samples=20000,
    # Opponent response model
    continue_prob_fn=logistic_continue_prob,  # maps R -> per-opponent continue prob
    # Rake
    rake_percent=0.0, rake_cap=None,
    # Your simulation module (for default equity)
    simulation_module=None,
):
    """
    Returns: dict with
      {
        'best_action': "fold"|"check"|"call"|"raise",
        'amount': 0 or raise size (over-the-call),
        'ev': best EV,
        'ev_call': EV(call),
        'details': {... breakdown ...}
      }
    Assumptions:
      - pot_value is the pot BEFORE your action.
      - call_value is what you must put in to call.
      - When you RAISE by R over the call:
          • You invest call_value + R (capped by hero_stack).
          • Each opponent independently continues with prob p(R).
          • If k opponents continue, each invests up to min(R, opp_stack).
      - If ALL fold (k=0), you win the current pot (no rake).
      - If k>=1, the hand “goes to showdown” now (rake applies) and equity vs k is used.
    """

    # ---- sanity / caps ----
    opponents = max(0, int(opponents))
    hero_stack = float('inf') if hero_stack is None else float(max(0, hero_stack))
    opp_stack  = float('inf') if opp_stack  is None else float(max(0, opp_stack))

    # Cap call and raises by stack
    max_affordable_raise = max(0, hero_stack - call_value)
    if max_raise <= 0:
        max_raise = 0
    if min_raise < 0:
        min_raise = 0
    # respect stack cap
    max_raise = min(max_raise, max_affordable_raise)

    # --- Open-raise sanity when there's nothing to call ---
    if call_value == 0:
        # Allow at most a standard open (~pot or 4x table min)
        max_raise = min(max_raise, pot_value, 4 * max(1, min_raise))

    # ---- default equity function (uses your simulation) ----
    if get_equity is None:
        if simulation_module is None:
            raise ValueError("Provide simulation_module or a get_equity(k) function.")
        @lru_cache(None)
        def _equity(k):
            # k = number of opponents that continue
            if k <= 0:
                return (1.0, 0.0)  # trivial: if no caller, you win pot uncontested
            win, lose, tie = simulation_module.monte_carlo(hand, board, opponents=k, samples=equity_samples)
            print(f"Simulation: {[win,lose,tie]}")
            # return (win, tie)
            return (win, tie)
        get_equity = _equity
    else:
        # cache user-supplied function too
        get_equity = lru_cache(None)(get_equity)

    # ---- EV helpers ----
    def apply_rake(pot_total):
        if rake_percent <= 0:
            return 0.0
        rake = pot_total * rake_percent
        if rake_cap is not None:
            rake = min(rake, rake_cap)
        return rake

    # EV of checking / calling
    def ev_call_option():
        if call_value == 0:
            # Pure check: no money goes in, pot stays; no one folds by assumption.
            # We go to next street/showdown only if that’s your model; here, treat as call of 0 with 1+ opponents “continuing”.
            # Use current opponents count to get equity.
            k = max(1, opponents)  # avoid trivial k=0 on a check
            win, tie = get_equity(k)
            pot_sd = pot_value  # no chips added
            rake = apply_rake(pot_sd)
            # Net: win*(pot - rake) + tie*0.5*(pot - rake) - lose*0
            lose = max(0.0, 1.0 - win - tie)
            return win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*0.0
        else:
            # Call: you invest call_value; assume no extra cold-callers (keep simple).
            # If you want cold-callers, model with a small p at R=0 and use binomial like the raise branch.
            k = max(1, opponents)
            win, tie = get_equity(k)
            lose = max(0.0, 1.0 - win - tie)
            pot_sd = pot_value + call_value + k*0.0  # only you add chips with a call in this simple branch
            # If you want others to also call current bet, add k*call_value above.
            rake = apply_rake(pot_sd)
            return win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*call_value

    # EV of a raise by R (over-the-call)
    def ev_raise_option(R):
        R = clamp(R, 0, max_raise)
        if R <= 0:
            return float('-inf')  # not a real raise
        # Opponent per-head continue prob for this R
        p_cont = clamp(continue_prob_fn(R), 0.0, 1.0)

        ev = 0.0
        # distribution of k callers among 'opponents'
        for k in range(0, opponents + 1):
            pk = binom_pmf(opponents, k, p_cont)
            if pk == 0.0:
                continue

            if k == 0:
                # Everyone folds: you win the pot; your bet comes back; no rake (typical).
                ev += pk * (pot_value)
                continue

            # At least one caller: showdown now.
            # Each caller can only call up to opp_stack, you can only invest up to hero_stack
            hero_invest = clamp(call_value + R, 0, hero_stack)
            caller_each = clamp(R, 0, opp_stack)
            total_callers_contrib = k * caller_each

            pot_sd = pot_value + hero_invest + total_callers_contrib
            rake = apply_rake(pot_sd)

            win, tie = get_equity(k)
            lose = max(0.0, 1.0 - win - tie)

            ev_k = win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*hero_invest
            ev += pk * ev_k

        return ev

    # ---- Evaluate actions ----
    # ---- Evaluate actions ----
    out_details = {}

    ev_call = ev_call_option()
    out_details['call' if call_value > 0 else 'check'] = ev_call

    best_action, best_amount, best_ev = ("check" if call_value == 0 else "call", 0, ev_call)

    # Gate: in open spots with many opponents, don't consider raises with trash equity
    allow_raises = True
    if call_value == 0 and opponents >= 3:
        equity_gate = 1.0 / (opponents + 1.5)  # e.g., vs 4 opp -> ~18.2%
        # Get equity for a typical called pot size (>=1 caller).
        # Use your sim for k = max(1, opponents) — cached in get_equity
        win_gate, tie_gate = get_equity(max(1, opponents))
        if win_gate < equity_gate:
            allow_raises = False

    # Scan raises only if allowed and sizes exist
    if allow_raises and min_raise > 0 and max_raise >= min_raise:
        R = min_raise
        seen = set()
        while R <= max_raise + 1e-9:
            Rr = round(R)
            if Rr not in seen:
                # Use the conservative continue prob
                def cont_prob_fn_wrapped(r):
                    return conservative_continue_prob(r, opponents)

                # Temporarily wrap the per-R continue prob
                p_cont = cont_prob_fn_wrapped(Rr)

                # Compute EV at this R using the existing ev_raise_option,
                # but feed the p_cont directly by briefly shadowing the fn:
                saved_fn = continue_prob_fn
                try:
                    # monkey-patch the closure variable used by ev_raise_option
                    def local_cp(_R, _pc=p_cont):
                        return _pc

                    # replace continue_prob_fn for this R
                    continue_prob_fn = local_cp  # type: ignore
                    evR = ev_raise_option(Rr)
                finally:
                    continue_prob_fn = saved_fn  # restore

                out_details[f'raise_{Rr}'] = evR
                if evR > best_ev:
                    best_action, best_amount, best_ev = ("raise", Rr, evR)
                seen.add(Rr)
            R += min_raise

    # Compare to fold if calling costs chips
    if call_value > 0 and best_ev <= 0.0:
        return {'best_action': 'fold', 'amount': 0, 'ev': 0.0, 'ev_call': ev_call, 'details': out_details}

    return {'best_action': best_action, 'amount': best_amount, 'ev': best_ev, 'ev_call': ev_call,
            'details': out_details}


def conservative_continue_prob(R, opponents):
    """
    Per-opponent continue probability vs raise size R (over-the-call).
    - Base logistic that drops with R.
    - With a floor that *increases* with number of opponents so multiway folds are rarer.
    """
    # Logistic: tune to your pool
    a, b = -0.2, -0.0002  # softer than before
    base = 1.0 / (1.0 + math.exp(-(a + b*float(R))))

    # Floor: at least some continue chance multiway.
    # Heads-up floor ~ 0.18, 3-4way ~ 0.28–0.32
    p_floor = 0.18 + 0.03 * max(0, opponents - 1)
    p_floor = min(p_floor, 0.32)
    return max(base, p_floor)


if __name__ == '__main__':
    game_window = pygetwindow.getWindowsWithTitle('GOP3')[0]

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

        key = cv.waitKey(10) & 0xFF

        if key == 32:
            # print("img saved")
            opponents = get_number_of_opponents(game_image)
            print(f"Number of opponents: {opponents}")

            left_hand = "".join(card for card in (classify_card_v2(left_card_image, 'left') or []) if card is not None)
            right_hand = "".join(card for card in (classify_card_v2(right_card_image, 'right') or []) if card is not None)
            flop1_card = "".join(card for card in (classify_card_v2(flop1_card_image, 'flop') or []) if card is not None)
            flop2_card = "".join(card for card in (classify_card_v2(flop2_card_image, 'flop') or []) if card is not None)
            flop3_card = "".join(card for card in (classify_card_v2(flop3_card_image, 'flop') or []) if card is not None)
            turn_card = "".join(card for card in (classify_card_v2(turn_card_image, 'flop') or []) if card is not None)
            river_card = "".join(card for card in (classify_card_v2(river_card_image, 'flop') or []) if card is not None)
            print(f"Player Cards: {left_hand}-{right_hand}")
            print(f"Flop1 cards: {flop1_card}")
            print(f"Flop2 cards: {flop2_card}")
            print(f"Flop3 cards: {flop3_card}")
            print(f"Turn cards: {turn_card}")
            print(f"River cards: {river_card}")
            #
            # win_prob, lose_prob, tie_prob = simulation.monte_carlo([left_hand, right_hand], [flop1_card, flop2_card, flop3_card, turn_card, river_card], opponents, samples=50000)
            # print(f"Simulation scores: {[win_prob, lose_prob, tie_prob]}")

            my_money = get_my_money(game_image)
            if my_money is None:
                my_money = 5000
            print(f"My money: {my_money}")
            pot_value = get_pot_value(game_image)
            print(f"Pot value: {pot_value}")
            call_value = get_call_value(game_image)
            print(f"Call Value: {call_value}")

            # expected_value = compute_ev(pot_value, call_value, win_prob, lose_prob + tie_prob)
            # print(f"EV(call) = {expected_value:.2f}")


            # After you have: hand, board, opponents, pot_value, call_value, my_money (hero stack), table_min_call, etc.

            # Effective stacks LEFT TO INVEST this street
            hero_stack_left = max(0, my_money)  # or your tracked stack
            opp_stack_left = 10_000_000  # if you don't know, set big; or detect

            result = decide_action_pro(
                hand=[left_hand, right_hand],
                board=[flop1_card, flop2_card, flop3_card, turn_card, river_card],
                opponents=opponents,
                pot_value=pot_value,
                call_value=call_value,  # 0 here → open spot
                min_raise=500,
                max_raise=5000,  # will be capped inside when call_value==0
                hero_stack=my_money,
                opp_stack=None,  # unknown → uncapped by opp
                simulation_module=simulation,
                equity_samples=50000,
                continue_prob_fn=conservative_continue_prob,  # the tamer curve above
                rake_percent=0.0,  # set if applicable
                rake_cap=None
            )

            print(f"EV(call/check) = {result['ev_call']:.2f}")
            if result['best_action'] == "raise":
                print(f"Best: RAISE {result['amount']} | EV = {result['ev']:.2f}")
            else:
                print(f"Best: {result['best_action'].upper()} | EV = {result['ev']:.2f}")

            print("------------------------")


