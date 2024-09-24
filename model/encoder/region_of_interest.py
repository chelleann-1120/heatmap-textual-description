import cv2
import numpy as np
import matplotlib.pyplot as plt


class ROI:
  '''
  Identifies the region of interest for the extraction of text
  '''

  def __init__(self, image_path):
    self.image_path = image_path
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
    max_area = 0
    largest_contour = None

    for contour in self.contours:
      area = cv2.contourArea(contour)

      if area > max_area:
        max_area = area
        largest_contour = contour

    return largest_contour
  
  def detect_color_legend(self):
  
    for contour in self.contours:
      area = cv2.contourArea(contour)

      if 1000 < area < 10000:
        return contour

  # Display the cropped region of interest
  def draw_bounding_box(self, roi):

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(gray_roi)
    cv2.rectangle(self.image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(self.image_np, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box")
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
    if self.largest_contour is not None:
      x, y, w, h = cv2.boundingRect(self.largest_contour)
      right = max(self.image_np.shape[1], x + 2 * w)
      legend_roi = self.image_np[y:y + h, x + w:right]
      self.draw_bounding_box(legend_roi)

      # Crop the image and return a copy without modifying the original
    return legend_roi