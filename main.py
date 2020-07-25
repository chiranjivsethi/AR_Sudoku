#Dependencies
import cv2
import imutils
import numpy as np

def preprocessing_image(img):
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) #kernel for dilation after adaptive thresholding
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Conver to Grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) #Denoise using gaussian blur of kernel (5,5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #adaptive threshold
    #morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #dilated = cv2.dilate(morph, kernel, iterations=1)
    return thresh

def get_sudoku_grid_contour(img):
    c, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(c, key=lambda x: cv2.contourArea(x), reverse=True)
    return c

def get_sudoku_grid_corners(c):
    contour = np.squeeze(c[0])

    sums = [sum(i) for i in contour]
    differences = [i[0] - i[1] for i in contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [contour[top_left], contour[top_right], contour[bottom_left],
               [bottom_right]]
    return corners

def warp_image(corners, img):
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    warped_image = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped_image

if __name__ == "__main__":
    #Loading Image
    test_image = cv2.imread('test_image.jpg')
    print("Image Loaded...")
    #cv2.imshow("Test Image",test_image)

    #Resizeing Image keeping same aspect ratio
    test_image = imutils.resize(test_image,width=500)
    print("Image resized...")
    #cv2.imshow("Resized Image keeping same aspect ratio",test_image)

    #Preprocessing Image
    processed_image = preprocessing_image(test_image)
    print("Image processed...")
    #cv2.imshow("Preprocessed Image",processed_image)

    img = test_image.copy()

    contour = get_sudoku_grid_contour(processed_image)
    cv2.drawContours(img, contour, 0, (0, 255, 0), 2)
    cv2.imshow("Sudoku Grid Contour",img)

    corners = get_sudoku_grid_corners(contour)

    #warped_image = warp_image(corners, processed_image)
    #cv2.imshow("Warped Image",warped_image)    

    cv2.waitKey(0)