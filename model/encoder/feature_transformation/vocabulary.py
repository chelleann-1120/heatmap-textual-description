import spacy
import re
from collections import Counter


class Vocabulary:
  def __init__(self, captions_list, freq_threshold=1):
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

    # Flatten the list of lists into a single list of captions
    # I won't need this though, nakaflatten na siya e
    flattened_captions = [caption for sublist in captions_list for caption in sublist]

    for caption in flattened_captions:
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

  def formatted_captions(self, captions_list):
    cleaned_caption = re.sub(r'([.,()])', r' \1 ', captions_list.lower())
    print(cleaned_caption)
    return cleaned_caption

  @property
  def size(self):
    return len(self.word_index) + 1

  @property
  def min_word_count(self):
    # get the word with the minimum count along with its count
    return min(self.word_counts.items(), key=lambda x: x[1])

  @property
  def max_word_count(self):
    return max(self.word_counts.items(), key=lambda x: x[1])

  def captions_to_sequence(self, sentence):
    if isinstance(sentence, list):
      sentence = " ".join(sentence)

    sentence = self.spacy_eng.tokenizer(sentence.lower())
    return [
        self.word_index[token.text]
        if token.text in self.word_index else self.word_index["<UNK>"]
        for token in sentence
    ]


  def sequence_to_captions(self, sequence):
      return " ".join([self.index_word[index] for index in sequence])

# Example usage

captions = [
   ["this is a sample caption! another sample caption"],
   ["another another, sample caption, because why not?"] 
]
vocab = Vocabulary(captions)
for i in captions:
  test = vocab.captions_to_sequence(i)
  print(test)
  print(vocab.sequence_to_captions(test))
  
