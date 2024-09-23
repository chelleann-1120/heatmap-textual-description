class TextFormatter:
    
  def __init__(self, text):
    self.text = text

  def format_title(self):
    title = self.text.extract_title()
    title = title.replace('\n', ' ')
    return title
  
  def clean_yaxis_label(self):
    
    yaxis_labels = self.text.extract_yaxis_labels()
    array_yaxis = yaxis_labels.splitlines()
    array_yaxis = [label for label in array_yaxis if label]
    
    return array_yaxis

  def clean_xaxis_label(self):

    xaxis_labels = self.text.extract_xaxis_labels()
    array_xaxis = xaxis_labels.splitlines()
    array_xaxis = [label for label in array_xaxis if label]
    
    return array_xaxis

  def clean_legend_values(self):

    legend_values = self.text.extract_legend_values()
    array_legend_values = legend_values.splitlines()
    array_legend_values = [value for value in array_legend_values if value]

    return array_legend_values
