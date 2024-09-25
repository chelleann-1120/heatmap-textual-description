from model.encoder.color_mapping import ColorMapping

class GridProcessor(ColorMapping):
  
  def __init__(self, text, image_path, image_name):
    super().__init__(text, image_path, image_name)
    self.title = text.format_title()
    self.yaxis_labels = text.clean_yaxis_label()
    self.xaxis_labels = text.clean_xaxis_label()
    self.legend_values = text.clean_legend_values()
    self.yaxis_len = len(self.yaxis_labels)
    self.xaxis_len = len(self.xaxis_labels)
    self.colors = self.extract_legend_color()
    self.grid_color = self.extract_grid_color()
    self.mapped_grid = self.map_cell_values()

  def create_grid_matrix(self):

    cell_matrix = []
    for j in range(self.yaxis_len):
      for i in range(self.xaxis_len):
        index = j * self.xaxis_len + i  # Calculate the index for 1D list
        mapped_value = self.mapped_grid[index]  # Access the 1D list element
        cell_matrix.append([self.yaxis_labels[j], self.xaxis_labels[i], mapped_value])
    return cell_matrix

  def flatten_list(self, nested_list):
    return [item for sublist in nested_list for item in sublist]