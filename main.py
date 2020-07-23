#Dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_sudoku_grid_contour(img):
    contours, hirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours

def process_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),1)
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    return img

def get_contour_corners(contours):
    contour = np.squeeze(contours[0])
    sums = [sum(i) for i in contour]
    differences = [i[0] - i[1] for i in contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [contour[top_left], contour[top_right], contour[bottom_left], contour[bottom_right]]
    return corners

def warp_image(corners,img):
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return img
'''
if __name__ == "__main__":
    
    Video = cv2.VideoCapture(0) 
  
    while(True): 
        ret, frame = Video.read()

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
    vid.release() 
    cv2.destroyAllWindows()
'''

if __name__ == "__main__":
    test_image  = cv2.imread('test_image.jpg')
    #cv2.imshow("Original",test_image)

    processed_image = process_image(test_image)
    #cv2.imshow("processed Image",processed_image)

    contours = get_sudoku_grid_contour(processed_image)
    cv2.drawContours(test_image, contours, 0, (0, 255, 0), 2)
    #cv2.imshow("Sudoku Grid Contour",test_image)

    corner = get_contour_corners(contours)
    warped_image = warp_image(corner,processed_image)
    #cv2.imshow("warped image",warped_image)

    cv2.waitKey(0)