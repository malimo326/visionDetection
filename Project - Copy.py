# import the necessary packages / libraries
import cv2
import os
import numpy as np

# Define the paths to the images and output directory
images_path='Bilder'
output_dir = 'Results'

# Function to detect boxes with errors in an image
def detect_boxes_with_errors(img_path):
    # Load the image
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # Apply thresholding to the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    
    # Find the contours in the image and print the number of contours detected
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours detected:", len(contours))

    # Loop through each contour
    for cnt in contours:
        x1,y1 = cnt[0][0] 
        # Find the approximate polygonal curve of the contour
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # condtions to detect the contour of boxes 
        if cv2.contourArea(cnt) > 20000 and cv2.contourArea(cnt) < 27500:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(h)/w
            print("w: ",w,"h: ",h)
            # Check if the contour ratio
            if ratio >= 0.9 and ratio <= 1.1:
                print("Contour Number:", cnt) # Print the contour values
                img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
                cv2.putText(img, 'Ok_box', (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'Error', (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                img = cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
    return img

# Loop through all the images in a folder
for filename in os.listdir(images_path):
    if filename.endswith('.jpg') or filename.endswith('.bmp'):
        image_path = os.path.join(images_path, filename)
        
        # Detect boxes with errors in the image
        result = detect_boxes_with_errors(image_path)
        
        # Save the image with the bounding boxes to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), result)

        # Display the result on screen
        cv2.imshow('Image', result)
        cv2.waitKey(3000) # Display the image for 3 seconds
        cv2.destroyAllWindows()
