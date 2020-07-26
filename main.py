import cv2
import imutils
import numpy as np

def preprocessing_image(img):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2, 2)
    )  # kernel for dilation after adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to Grayscale
    blur = cv2.GaussianBlur(
        gray, (5, 5), 0
    )  # Denoise using gaussian blur of kernel (5,5)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )  # adaptive threshold
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    return thresh

def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners


def warp_perspective(pts, img):  # TODO: Spline transform, remove this
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped


if __name__ == "__main__":
    # Loading Image
    test_image = cv2.imread("test_image.jpg")
    print("Image Loaded...")
    # cv2.imshow("Test Image",test_image)

    # Resizeing Image keeping same aspect ratio
    test_image = imutils.resize(test_image, width=500)
    print("Image resized...")
    # cv2.imshow("Resized Image keeping same aspect ratio",test_image)

    # Preprocessing Image
    processed_image = preprocessing_image(test_image)
    print("Image processed...")
    # cv2.imshow("Preprocessed Image",processed_image)

    corners = get_corners(processed_image)
    print("Sudoku grid contour corners extracted...")

    warped_image = warp_perspective(corners, processed_image)
    print("Image warped...")
    #cv2.imshow("Warped Image",warped_image)

    

    cv2.waitKey(0)