import tensorflow as tf
import numpy as np


class EmbeddingLayer:

    def __init__(self, matrix_values):
        # Hashing layer and embeddings for x axis and legend range
        self.matrix_values = matrix_values
        self.hashing_layer = tf.keras.layers.Hashing(num_bins=1000)

    def get_yaxis_embedding(self, yaxis_label):
        yaxis_idx = self.hashing_layer(tf.convert_to_tensor(yaxis_label, dtype=tf.string))
        return self.get_embedding_layer(yaxis_idx)

    def get_legend_embedding(self, legend_values):
        legend_idx = self.hashing_layer(tf.convert_to_tensor(legend_values, dtype=tf.string))
        return self.get_embedding_layer(legend_idx)
    
    def get_xaxis_embedding(self, xaxis_label):
        xaxis_idx = self.hashing_layer(tf.convert_to_tensor(xaxis_label, dtype=tf.string))
        return self.get_embedding_layer(xaxis_idx)
    
    def get_embedding_layer(self, input):
        embedding_layer = tf.keras.layers.Embedding(input_dim=1000, output_dim=5)
        return embedding_layer(input)

    def preprocess_features(self):
        #rgb_vectors = []
        legend_val_vector = 0
        grouped_vectors = []

        for x_value, y_value, legend_value in self.matrix_values:
            # Process country
            y_value_vector = self.get_yaxis_embedding(y_value)
            x_value_vector = self.get_xaxis_embedding(x_value)

            # Process legend range using hashing and embedding
            if legend_value != 'NaN':
                legend_vector = self.get_legend_embedding(legend_value)
                legend_val_vector = tf.reshape(legend_vector, [-1])
            else:
                # Handle NaN or invalid legend range
                legend_val_vector = tf.zeros(5)

            combined_vector = tf.stack([x_value_vector, y_value_vector, legend_val_vector], axis=1)
            grouped_vectors.append(combined_vector)

        grouped_vectors_3d = tf.convert_to_tensor(grouped_vectors)
        return grouped_vectors_3d