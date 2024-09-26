import numpy as np
import cv2
from PIL import Image
from model.encoder.feature_extraction.region_detection import RegionDetection

class ColorExtractor(RegionDetection):
  '''
  Extracts color in the given image contour
  '''

  def __init__(self, image_path, image_name):
    super().__init__(image_path, image_name)
    self.image_path = image_path
    self.color_legend = self.detect_color_legend()

  def extract_legend_color(self):
    image = Image.open(self.image_path).convert('RGB')
    image_array = np.array(image)

    x, y, width, height = cv2.boundingRect(self.color_legend)
    legend_roi = image_array[y:y+height, x:x+width]

    # iterates over the pixels of the given image
    flattened_colors = [color for row in legend_roi for color in row if not np.array_equal(color, [0, 0, 0])]

    unique_colors = []
    seen = set()

    for color in flattened_colors:
      color_tuple = tuple(color)

      if color_tuple not in seen:
        seen.add(color_tuple)
        unique_colors.append(color)

    return unique_colors
  
  def extract_grid_color(self):
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(self.largest_contour)

    # Crop the region of interest from the original image using the bounding box
    cropped_image = self.image[y:y+h, x:x+w]

    num_regions_y = self.yaxis_len
    num_regions_x = self.xaxis_len

    # Calculate the new dimensions that are divisible by the specified number of regions
    new_height = (h // num_regions_y) * num_regions_y
    new_width = (w // num_regions_x) * num_regions_x

    # Resize the cropped image to the new dimensions while maintaining the aspect ratio
    resized_image = cv2.resize(cropped_image, (new_width, new_height))
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Calculate the width and height of each smaller region
    region_height = new_height // num_regions_y
    region_width = new_width // num_regions_x

    dominant_colors = []

    # Extract and display each smaller region
    for i in range(num_regions_y):
      for j in range(num_regions_x):
          
        start_y = i * region_height
        end_y = start_y + region_height
        start_x = j * region_width
        end_x = start_x + region_width
        smaller_region = resized_image_rgb[start_y:end_y, start_x:end_x]

        # Convert the smaller region to a 2D array of pixels
        pixels = smaller_region.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Define criteria and apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 1
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Get the dominant color
        dominant_color = centers[0].astype(int).tolist()
        dominant_colors.append(dominant_color)

    return dominant_colors