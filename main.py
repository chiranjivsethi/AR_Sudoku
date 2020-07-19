import cv2 

def process_image(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),1)
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    return img

def sudoku_contour(img):
    contours, hirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours

if __name__ == "__main__":
    
    Video = cv2.VideoCapture(0) 
  
    while(True): 
        ret, frame = Video.read() 
        cv2.imshow('frame', frame) 

        processed_image = process_image(frame)
        cv2.imshow("Processed Image",processed_image)

        contours = sudoku_contour(processed_image)
        cv2.drawContours(frame, contours, 0, (0, 255, 0), 2)
        cv2.imshow("Grid Sudoku Constour",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
    vid.release() 
    cv2.destroyAllWindows() 