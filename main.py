import cv2
import numpy as np

image_paths = ['foto_sardegna_24.png']  # Add your image paths here

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges 
    edged = cv2.Canny(gray, 100, 200, L2gradient=True, apertureSize=7) 
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    cropped_images = []

    # Minumum size for the image rectangle
    min_size = (image.shape[0] * image.shape[1])**0.5 * 0.1
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour has 4 points (rectangle)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > min_size and h > min_size:  # Filter out small contours
                # Get the perspective transform matrix
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                width = int(rect[1][0])
                height = int(rect[1][1])
                
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image, M, (width, height))
                
                cropped_images.append(warped)
                
                # Draw rectangle around detected photo
                cv2.drawContours(image, [box], 0, (0, 255, 0), int(min_size/50))
                cv2.drawContours(edged, [box], 0, (0, 255, 0), int(min_size/50))
    
    return image, cropped_images, edged



for image_path in image_paths:
    processed_image, cropped_images, edged_image = preprocess_image(image_path)
    
    # Scale down the image for display
    scale_percent = 5  # percent of original size
    width = 800
    height = int(processed_image.shape[0] * width / processed_image.shape[1])
    dim = (width, height)
    resized_image = cv2.resize(processed_image, dim, interpolation=cv2.INTER_AREA)
    resized_edged = cv2.resize(edged_image, dim, interpolation=cv2.INTER_AREA)
    
    # Save or display the processed image with rectangles
    cv2.imwrite(f'processed_{image_path}', processed_image)
    cv2.imshow(f'Processed {image_path}', resized_image)
    cv2.imshow(f'Edged {image_path}', resized_edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally, save the cropped images
    for i, cropped in enumerate(cropped_images):
        cv2.imwrite(f'cropped_{i}_{image_path}', cropped)