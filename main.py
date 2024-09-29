import os, sys
from model.encoder.feature_transformation.data_transformation import DataTransformation, Vocabulary


class HeatmapEncoderDecoder:

  def __init__(self, input_dir):
    self.train_csv_path = ".\\data\\ground_truth\\train\\annotations.csv"
    self.train_img_dir = ".\\data\\images\\train"
    self.max_len = 0


  def run_main(self):

    data = DataTransformation(self.train_csv_path, self.train_img_dir)
    vocab = Vocabulary(data, freq_threshold=2)

    # Training

    # Save model

    # Testing

    # Validation

if __name__ == "__main__":
  input_dir = ".\\data\\images\\train"
  HeatmapEncoderDecoder(input_dir).run_main()