import os, sys
from model.preprocessor.data_preprocessing import DataPreprocessing, Vocabulary


class HeatmapEncoder:

  def __init__(self, input_dir):
    self.train_csv_path = ".\\data\\ground_truth\\train\\annotation.csv"
    self.train_img_dir = "\\data\\images\\train"
    self.max_len = 0


  def run_main(self):
    
    data = DataPreprocessing(self.train_csv_path, self.train_img_dir).prepare_annotation
    vocab = Vocabulary(data, freq_threshold=4)

    # Data Preprocessing
    # vocab = Vocabulary(data, freq_threshold=2)

    # Training

    # Save every epoch or after the training?

    # Testing

if __name__ == "__main__":
  input_dir = ".\\data\\images\\train"
  HeatmapEncoder(input_dir).run_main()