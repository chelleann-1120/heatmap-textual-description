from model.encoder.text_extraction import TextExtraction
from model.encoder.format_text import TextFormatter
from model.encoder.grid_processor import GridProcessor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class DataPreprocessing:
  '''
  Extracts the texts in each heatmap
  '''

  def __init__(self, input_dir):
    self.input_dir = input_dir
    self.images = os.listdir(input_dir)
    self.encoder_input = LabelEncoder() # mali
    self.encoder_target = LabelEncoder() # mali
    self.max_len = 0

  @property
  def get_max_length(self):
    max_input_len = self.max_length_input
    max_len_target = self.max_len_target()
    return max_input_len, max_len_target


  @property
  def max_length_input(self):

    for image in self.images[:1]:
      length = self.process_image(image)

      if length > self.max_len:
        self.max_len = length
        print(self.max_len)
    
    return self.max_len
  
  def process_image(self, image):
    image_path = os.path.join(self.input_dir, image)

    try:
      values = TextExtraction(image_path, image)
      clean_values = TextFormatter(values)
      matrix_values = GridProcessor(clean_values, image_path, image).create_grid_matrix()
      
      return len(matrix_values)

    except Exception as e:
      print(f"Error processing image {image}: {e}")
      return 0
    
  def process_target(self):
    pass
  
  def max_len_target(self):
    pass
  
  
  def tokenize_input(self):
    pass


class EncodeText:
  '''
  Encodes the input text and adds a padding in a specified maximum length
  '''

  # def __init__(self):
  #   self.input_max_len = DataPreprocessing.find_max_length()
  #   self.target_max_len = DataPreprocessing.find_max_len_target()


  def encode_text(self):
    pass

  # Encode and pad the ground truth
  def encode_pad_target(self, target_output):

    target_tokens = target_output.lower().split()
    target_tokens = ['<SOS>'] + target_tokens + ['<EOS>']

    self.encoder_target.fit(target_tokens)
    encoded_target_tokens = self.encoder_target.transform(target_tokens)

    padded_target_tokens = pad_sequences([encoded_target_tokens], maxlen=self.target_max_len, padding='post', value=self.encoder_target.transform(['<PAD>'])[0])

    return padded_target_tokens

  # Encode and pad the extracted text from images
  def encode_pad_input(self, input_data):
    encoded_padded_data = []

    for sublist in input_data:

      tokens = [str(item).lower() for item in sublist]
      tokens = ['<SOS>'] + tokens + ['<EOS>']

      self.encoder_input.fit(tokens)
      encoded_tokens = self.encoder_input.transform(tokens)

      padded_tokens = pad_sequences([encoded_tokens], maxlen=self.input_max_len, padding='post', value=self.encoder_input.transform(['<PAD>'])[0])
      encoded_padded_data.append(padded_tokens[0])

    return np.array(encoded_padded_data)


# iterate through all the text length, find the max length
# iterate through all the text again and padd
# each iterated item padded will go through the embedding layer