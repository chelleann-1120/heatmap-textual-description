import cv2
import numpy as np
import matplotlib.pyplot as plt


class RegionDetection:
  '''
  Identifies the region of interest for the extraction of text
  '''

  def __init__(self, image_path, image_name):
    self.image_path = image_path
    self.image_name = image_name
    self.image = cv2.imread(self.image_path)
    self.contours = self.find_contours()
    self.image_np = np.array(self.image)
    self.largest_contour = self.detect_grid()

  def find_contours(self):

    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

  def detect_grid(self):

    largest_contour = max(self.contours, key=cv2.contourArea)
    return largest_contour
  
  def detect_color_legend(self):
  
    # Calculate areas for all contours
    contours_with_areas = [(contour, cv2.contourArea(contour)) for contour in self.contours]
    
    # Sort contours by area in descending order
    sorted_contours = sorted(contours_with_areas, key=lambda x: x[1], reverse=True)
    
    # Return the second largest contour if it exists
    if len(sorted_contours) > 1:
      second_largest_contour = sorted_contours[1][0]
      return second_largest_contour

  # Displays the region of interest
  def crop_detected_roi(self, roi):

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate the bounding box coordinates
    x, y, w, h = cv2.boundingRect(gray_roi)
    
    # Debug: Print the coordinates
    print(f"Bounding Box Coordinates: x={x}, y={y}, w={w}, h={h}")
    
    # Draw the bounding box on the original image
    cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Debug: Display the grayscale ROI
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


  def detect_title_roi(self):
    
    if self.largest_contour is not None:
      x, y, w, h = cv2.boundingRect(self.largest_contour)
      top = max(0, y - h)
      title_roi = self.image_np[top:y, :]

      return title_roi
  
  def detect_yaxis_roi(self):

    if self.largest_contour is not None:
      x, y, w, h = cv2.boundingRect(self.largest_contour)
      left = max(0, x - w)
      yaxis_roi = self.image_np[y:y + h, left:x]

      return yaxis_roi

  def detect_xaxis_roi(self):

    if self.largest_contour is not None:
      x, y, w, h = cv2.boundingRect(self.largest_contour)
      bottom = min(self.image_np.shape[0], y + 2 * h)
      xaxis_roi = self.image_np[y + h:bottom, x:x + w]

      return xaxis_roi

  def detect_legend_roi(self):
    
    x, y, w, h = cv2.boundingRect(self.largest_contour)
    right = max(self.image.shape[1], x + 2 * w)
    legend_roi = self.image[y:y + h, x + w:right]

    return legend_roi