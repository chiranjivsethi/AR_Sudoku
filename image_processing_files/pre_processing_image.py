import cv2
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocesing")
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    args = vars(parser.parse_args())

    image = cv2.imread(args["input"])
    image = preprocessing_image(image)
    cv2.imshow("Pre Processed Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

