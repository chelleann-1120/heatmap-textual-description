from model.encoder.text_extraction import TextExtraction
from model.encoder.format_text import TextFormatter
from model.encoder.grid_processor import GridProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
import re
from collections import Counter
import os
import re

# I'll think of a better name
class DataPreprocessing:
  '''
  Converts the annotation and extracted texts from images into tokens or sequence
  '''
  def __init__(self, csv_dir, img_dir):
    self.csv_dir = csv_dir
    self.img_dir = img_dir

  def read_csv(self, file_path):
    df = pd.read_csv(file_path)
    return df
  
  @property
  def preprocess_data(self):
    all_img_data, all_annotation_data = self.process_all_data
    print(all_annotation_data)

  @property
  def process_image(self):
    img_data = []
    annotation_data = []
    df = self.read_csv(self.csv_dir)

    for _, row in df.iloc[0:2].iterrows():
      image_name = row.iloc[0]
      image_path = os.path.join(self.img_dir, image_name)

      values = TextExtraction(image_path, image_name)
      clean_values = TextFormatter(values)
      matrix_values = GridProcessor(clean_values, image_path, image_name).create_grid_matrix()
      img_data.extend(matrix_values)
      
      annotation = self.formatted_captions(row.iloc[1])
      annotation_data.append(annotation)

    return img_data, annotation_data

  def words_to_sequence(self, sentence):

    tokens = self.spacy_eng.tokenizer(sentence.lower())
    return [
        self.word_index[token.text]
        if token.text in self.word_index else self.word_index["<UNK>"]
        for token in tokens
    ]

  # def convert_range(self): Will implement separately in image processing part of code
  # pass

  def formatted_captions(self, captions):
    cleaned_caption = re.sub(r'([.,()])', r' \1 ', captions.lower())
    return self.tokenize_target(cleaned_caption)

  def tokenize_target(self, annotation):
    return word_tokenize(annotation)

  def tokenize_input(self, matrix_values):
    tokenized_words = [[word for item in sub_list for word in word_tokenize(item)] for sub_list in matrix_values]
    return tokenized_words

  @property
  def prepare_annotation(self):
    annotation_data = []
    df = self.read_csv(self.csv_dir)

    for _, row in df.iloc[0:2].iterrows():
      annotation = self.formatted_captions(row.iloc[1])
      annotation_data.extend(annotation)
    
    return annotation_data
  

class Vocabulary:

  def __init__(self, captions_list, freq_threshold=2):
    self.word_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.index_word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.word_counts = {}
    self.freq_threshold = freq_threshold
    spacy.prefer_gpu()
    self.spacy_eng = spacy.load("en_core_web_sm")
    self.build_vocab(captions_list)

  def build_vocab(self, captions_list):
    frequency = Counter()
    index = len(self.word_index)

    for caption in captions_list:
      print(caption)
      tokens = self.spacy_eng.tokenizer(caption)
      for token in tokens:
        frequency[token.text] += 1

        if frequency[token.text] == self.freq_threshold:
          self.word_index[token.text] = index
          self.index_word[index] = f"{token.text}"
          index += 1

    print(self.word_index)
    print(self.index_word)
  
  # Is used in a loop, input is a list
  def captions_to_sequence(self, sentence):
    if isinstance(sentence, list):
      sentence = " ".join(sentence)

    sentence = self.spacy_eng.tokenizer(sentence.lower())
    return [
        self.word_index[token.text]
        if token.text in self.word_index else self.word_index["<UNK>"]
        for token in sentence
    ]

  # Is used in a loop, input is a list
  def sequence_to_captions(self, sequence):
      return " ".join([self.index_word[index] for index in sequence])
  