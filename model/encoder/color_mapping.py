import numpy as np
from model.encoder.color_extraction import ColorExtractor

class ColorMapping(ColorExtractor):

  def __init__(self, text, image_path, image_name):
    super().__init__(image_path, image_name)
    self.title = text.format_title()
    self.yaxis_labels = text.clean_yaxis_label()
    self.xaxis_labels = text.clean_xaxis_label()
    self.legend_values = text.clean_legend_values()
    self.yaxis_len = len(self.yaxis_labels)
    self.xaxis_len = len(self.xaxis_labels)
    self.colors = self.extract_legend_color()
    self.grid_colors = self.extract_grid_color()


  def map_legend_values(self):
    title = self.legend_values[0]
    legend_dict = {"title": title}
    legend_dict.update({tuple(self.colors[i]): self.legend_values[i + 1] for i in range(len(self.colors))})
    return legend_dict

  def color_distance(self, color, legend_color):
    return np.linalg.norm(np.array(color) - np.array(legend_color))

  def find_closest_color(self, color, legend_dict):
    closest_color = None
    min_distance = float('inf')

    for legend_color in legend_dict.keys():
      if legend_color == "title":
        continue
      distance = self.color_distance(color, legend_color)
      if distance < min_distance:
        min_distance = distance
        closest_color = legend_color
    return closest_color

  def map_cell_values(self):
    legend_dict = self.map_legend_values()
    mapped_legend_values = []

    for color in self.grid_colors:

      color_tuple = tuple(color)  # Convert array to tuple for dictionary key
      closest_color = self.find_closest_color(color_tuple, legend_dict)

      if closest_color is not None:
        # mapped_legend_values.append([closest_color, legend_dict[closest_color]])
        mapped_legend_values.append(legend_dict[closest_color])
      else:
        mapped_legend_values.append((color_tuple, None))  # Handle case where no close color is found
    
    return mapped_legend_values