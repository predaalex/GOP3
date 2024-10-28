import cv2 as cv
import pygetwindow
import os

import cv_functions

save_dir = os.path.join(os.curdir, "cards")

os.makedirs(save_dir, exist_ok=True)

game_window = pygetwindow.getWindowsWithTitle('GOP3')[1]
counter = 0
while True:
    game_image = cv_functions.get_image(game_window)
    cv.imshow("screenshot", game_image)
    key = cv.waitKey(1) & 0xFF

    if key == 32:
        print("Space: ScreenShot!")
        # Get img
        game_image = cv_functions.get_image(game_window)

        # Get each possible card
        left_card = cv_functions.extract_image(game_image, cv_functions.left_card_coords)
        right_card = cv_functions.extract_image(game_image, cv_functions.right_card_coords)
        flop1_card = cv_functions.extract_image(game_image, cv_functions.flop1)
        flop2_card = cv_functions.extract_image(game_image, cv_functions.flop2)
        flop3_card = cv_functions.extract_image(game_image, cv_functions.flop3)
        turn_card = cv_functions.extract_image(game_image, cv_functions.turn)
        river_card = cv_functions.extract_image(game_image, cv_functions.river)

        # Position left, right, or flop (l, r, or f)
        print("Press l, r, or f for position left, right, or flop")
        position_key = cv.waitKey(0) & 0xFF
        flop_position = ""
        if position_key == ord('l'):
            position = "left"
        elif position_key == ord('r'):
            position = "right"
        else:
            position = "flop"
            print(f"Press 1, 2, 3, 4, or 4 for flop position")
            flop_position = cv.waitKey(0) & 0xFF
            flop_position = chr(flop_position)

        print(f"Position: {position}{flop_position}")

        # Get rank
        print("Press h for hearth | c for clubs | d for diamonds | s for stripes")
        rank = chr(cv.waitKey(0) & 0xFF)
        print(f"Rank: {rank}")

        # Get value
        print("Press for value 1, 2, 3... 9, T, J, Q, K, A")
        value = chr(cv.waitKey(0) & 0xFF)
        print(f"Value: {value}")

        # Construct card name and path
        card_name = f"{rank}{value}"
        position_dir = os.path.join(save_dir, position)

        # Ensure the directory for the position exists
        os.makedirs(position_dir, exist_ok=True)

        card_path = os.path.join(position_dir, f"{card_name}.png")

        # Check if the card already exists, then save
        if not os.path.exists(card_path):

            save_img = game_image

            if position == "flop":
                if flop_position == "1":
                    save_img = flop1_card
                elif flop_position == "2":
                    save_img = flop2_card
                elif flop_position == "3":
                    save_img = flop3_card
                elif flop_position == "4":
                    save_img = turn_card
                elif flop_position == "5":
                    save_img = river_card
                else:
                    print("ce pla mea")
            elif position == "left":
                save_img = left_card
            elif position == "right":
                save_img = right_card
            else:
                print("Wrong position")
                continue

            cv.imwrite(card_path, save_img)
            print(f"Saved: {card_path}")
        else:
            print(f"Card {card_name} already exists in {position}")
    elif key == ord('q'):
        cv.destroyAllWindows()
        exit(0)
