import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract


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


class ColorExtractor(ROI):
  '''
  Extracts color in the given image contour
  '''

  def __init__(self, image_path):
    super().__init__(image_path)
    self.image_path = image_path
    self.color_legend = self.detect_color_legend()


class TextExtraction(ColorExtractor):
  '''
  Extracts the text in the given region of interest.
  '''
  def __init__(self, image_path):
    super().__init__(image_path)
    self.image_path = image_path
    self.img = cv2.imread(image_path)
    #self.legend_color = self.extract_legend_color()
    self.title_roi = self.detect_title_roi()
    self.yaxis_roi = self.detect_yaxis_roi()
    self.xaxis_roi = self.detect_xaxis_roi()
    self.legend_roi = self.detect_legend_roi()

  def extract_title(self):
    title_text = pytesseract.image_to_string(self.title_roi)
    print("Extracted title text:", title_text)
    return title_text

  def extract_yaxis_labels(self):
    yaxis_text = pytesseract.image_to_string(self.yaxis_roi)
    print("Extracted y-axis labels text:", yaxis_text)
    return yaxis_text

  def extract_xaxis_labels(self):
    
    xaxis_roi = self.detect_xaxis_roi()
    
    cropped_image = Image.fromarray(cv2.cvtColor(xaxis_roi, cv2.COLOR_BGR2RGB))
    rotated_image = cropped_image.rotate(-90, expand=True)
    
    xaxis_text = pytesseract.image_to_string(rotated_image)
    print("Extracted x-axis labels text:", xaxis_text)
    return xaxis_text

  def remove_legend(self):

    contour = self.detect_color_legend()
    if contour is not None:
        cv2.drawContours(self.image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    else:
        print("No legend contour found")


  def extract_legend_values(self):
    self.remove_legend()

    largest_contour = self.detect_grid()
    if largest_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = self.image_np[y:y+h, x+w:]

    # Convert the ROI to a PIL image
    roi_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Extract text using Tesseract
    legend_text = pytesseract.image_to_string(roi_image)
    print("Extracted legend text:", legend_text)
    return legend_text


  def display_image(self):
    plt.imshow(cv2.cvtColor(self.image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()


class GridProcessor:
  pass


class ColorMapping(TextExtraction, GridProcessor):
  
  def __init__(self, image_path):
    super.__init__(self, image_path)
    pass

  def map_legend_values(self):
    pass

  def map_gridcells_values(self):
    pass
  
