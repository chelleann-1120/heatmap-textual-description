from model.encoder.feature_extraction.text_extraction import TextExtraction
from model.encoder.feature_extraction.format_text import TextFormatter
from model.encoder.feature_extraction.grid_processor import GridProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
import spacy
import re
from collections import Counter
import os
import re


class DataPreprocessing:
  '''
  Converts the annotation and extracted texts from images into tokens or sequence
  '''
  def __init__(self, csv_dir, img_dir):
    self.csv_dir = csv_dir
    self.img_dir = img_dir
    spacy.prefer_gpu()
    self.spacy_eng = spacy.load("en_core_web_sm")


  def read_csv(self, file_path):
    df = pd.read_csv(file_path)
    return df
  
  @property
  def preprocess_data(self):
    all_img_data, all_annotation_data = self.collect_data
    return all_img_data, all_annotation_data

  @property
  def collect_data(self): # Returns tokenized img data, and annotation data
    img_data = []
    annotation_data = []
    df = self.read_csv(self.csv_dir)

    for _, row in df.iloc[:2].iterrows():
      image_name = row.iloc[0]
      image_path = os.path.join(self.img_dir, image_name)

      values = TextExtraction(image_path, image_name)
      clean_values = TextFormatter(values)
      
      matrix_values = GridProcessor(clean_values, image_path, image_name).create_grid_matrix()
      img_data.append(self.tokenize_input(matrix_values))
      
      annotation = self.formatted_captions(row.iloc[1])
      tokens = self.spacy_eng.tokenizer(annotation)
      tokenized_annotation = [token.text for token in tokens]
      annotation_data.append(tokenized_annotation)

    return img_data, annotation_data

  def words_to_sequence(self, sentence):

    tokens = self.spacy_eng.tokenizer(sentence.lower())
    return [
        self.word_index[token.text]
        if token.text in self.word_index else self.word_index["<UNK>"]
        for token in tokens
    ]

  def formatted_captions(self, captions):
    cleaned_caption = re.sub(r'([.,()])', r' \1 ', captions.lower())
    return cleaned_caption

  def tokenize_input(self, matrix_values):
    tokenized_matrix = [
        [word.text for item in sub_list for word in self.spacy_eng.tokenizer(str(item))]
        for sub_list in matrix_values
    ]
    return tokenized_matrix

  
  def tokenize_annotation(self, annotation_list):
    annotation_list_tokens = []
    
    for annotation in annotation_list:
      if isinstance(annotation, list):
          annotation = " ".join(annotation)

      tokens = [token.text for token in self.spacy_eng.tokenizer(annotation)]
      annotation_list_tokens.append(tokens)
    
    return annotation_list_tokens

  @property
  def prepare_annotation(self):
    annotation_data = []
    df = self.read_csv(self.csv_dir)

    for _, row in df.iloc[0:2].iterrows():
      annotation = self.formatted_captions(row.iloc[1])
      annotation_data.extend(self.spacy_eng.tokenizer(annotation))
    
    return annotation_data


class Vocabulary:
  '''
  Converts the tokenized texts into equivalent sequence in dictionary
  '''
  def __init__(self, data, freq_threshold=2):
    spacy.prefer_gpu()
    self.word_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.index_word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.word_counts = {}
    self.freq_threshold = freq_threshold
    self.annotation_sequence = []
    self.img_text_sequence = []
    self.spacy_eng = spacy.load("en_core_web_sm")
    self.data = data
    self.build_vocab(self.data.prepare_annotation)
    self.convert_to_sequence(self.data.preprocess_data)
    self.pad_data()
    

  def build_vocab(self, captions_list):
    frequency = Counter()
    index = len(self.word_index)

    for token in captions_list:
      frequency[token.text] += 1

      if frequency[token.text] == self.freq_threshold:
        self.word_index[token.text] = index
        self.index_word[index] = f"{token.text}"
        index += 1

    print(self.index_word)
  
  
  # Is used in a loop, input is a single list
  def annotation_to_sequence(self, sentence):
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    # Convert sentence to a list of tokens
    tokens = sentence.split()

    # Convert tokens to indices
    sequence = [
        self.word_index[token]
        if token in self.word_index else self.word_index["<UNK>"]
        for token in tokens
    ]

    # Append <SOS> at the start and <EOS> at the end
    sequence.insert(0, self.word_index["<SOS>"])
    sequence.append(self.word_index["<EOS>"])

    return sequence
  
  def img_text_to_seq(self, list):
    return [self.annotation_to_sequence(sublist) for sublist in list]

  # Is used in a loop, input is a single list
  def sequence_to_captions(self, sequence):
      return " ".join([self.index_word[index] for index in sequence])
  
  def convert_to_sequence(self, collected_data):
    img_text_list, annotation_list = collected_data
    all_img_text_seq = []
    all_annotation_seq = []

    for img_text in img_text_list:
      img_txt_seq = self.img_text_to_seq(img_text)
      img_data = tf.keras.preprocessing.sequence.pad_sequences(
                  img_txt_seq, maxlen=8, padding="post"
                  )
      all_img_text_seq.append(img_data)

    for annotation in annotation_list:
      annotation_sequence = self.annotation_to_sequence(annotation)
      all_annotation_seq.append(annotation_sequence)
      # print(annotation_sequence)
    
    return all_img_text_seq, all_annotation_seq
  
  def pad_data(self):

    img_data, annotation_data = self.convert_to_sequence(self.data.preprocess_data)
    
    annotation_data = tf.keras.preprocessing.sequence.pad_sequences(
        annotation_data, padding="post")

    img_data = tf.keras.preprocessing.sequence.pad_sequences(
                  img_data, maxlen=565, padding="post"
                  )

    print(annotation_data)

    return np.array(annotation_data)
    

  

    
