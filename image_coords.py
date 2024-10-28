import cv2


# Callback function to handle mouse events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates
        coords.append(f"({x}, {y}), ")
        if len(coords) == 2:
            output = ""
            for coord in coords:
                output += coord
            print(output[:-2])
        elif len(coords) >= 3:
            coords.clear()
            coords.append(f"({x}, {y}),")

        # Draw a small circle at the clicked point (for visual feedback)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        # Display the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"{x}, {y}", (x + 10, y + 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Show the updated image
        cv2.imshow('image', image)


# Read the image
image_path = 'resources/second.png'
image = cv2.imread(image_path)
coords = []

if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Display the image in a window
    cv2.imshow('image', image)

    # Set the mouse callback function to handle clicks
    cv2.setMouseCallback('image', click_event)

    # Wait for the user to press a key
    cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
