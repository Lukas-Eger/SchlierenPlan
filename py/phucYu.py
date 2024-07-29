import cv2
import numpy as np



def imageCropping(image):
    
    global x_start, y_start, x_end, y_end, cropping
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    oriImage = image.copy()

    def mouse_crop(event, x, y, flags, param):
        global x_start, y_start, x_end, y_end, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
            refPoint = [(x_start, y_start), (x_end, y_end)]
            if len(refPoint) == 2:
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)

    cv2.destroyAllWindows()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to exit the loop
            break

    cv2.destroyAllWindows()
    return oriImage[y_start:y_end, x_start:x_end]

img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
cropped_img = imageCropping(img)
if cropped_img is not None:
    cv2.imshow("Final Cropped Image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
