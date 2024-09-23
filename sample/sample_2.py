import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Attention
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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
    
    def get_title_embedding(self, xaxis_label):
      title_idx = self.hashing_layer(tf.convert_to_tensor(xaxis_label, dtype=tf.string))
      return self.get_embedding_layer(title_idx)
    
    def get_embedding_layer(self, input):
      embedding_layer = tf.keras.layers.Embedding(input_dim=1000, output_dim=16)
      return embedding_layer(input)

    def preprocess_features(self):  
      grouped_vectors = []

      for x_value, y_value, legend_value, title_value in self.matrix_values:
        y_value_vector = self.get_yaxis_embedding(y_value)
        x_value_vector = self.get_xaxis_embedding(x_value)
        title_vector = self.get_title_embedding(title_value)

        if legend_value != 'NaN':
            legend_vector = self.get_legend_embedding(legend_value)
            legend_val_vector = tf.reshape(legend_vector, [-1])
        else:
            legend_val_vector = tf.zeros(16)

        combined_vector = tf.stack([x_value_vector, y_value_vector, legend_val_vector, title_vector], axis=1)
        grouped_vectors.append(combined_vector)

      grouped_vectors_3d = tf.convert_to_tensor(grouped_vectors)
      return grouped_vectors_3d

# Example matrix_values
matrix_values = [
    ['Sierra Leone', '1999.2001', 'NaN', 'corruption'], 
    ['Sierra Leone', '2002.2003', 'NaN', 'corruption'], 
    ['Sierra Leone', '2004.2005', 'NaN', 'corruption'], 
    ['Sierra Leone', '2006.2007', 'NaN', 'corruption'], 
    ['Sierra Leone', '2008.2009', 'NaN', 'corruption'], 
    ['Sierra Leone', '2010.2011', 'NaN', 'corruption'], 
    ['Sierra Leone', '2012.2013', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption']
]

test_values = [
    ['Sierra Leone', '1999.2001', 'NaN', 'corruption'], 
    ['Sierra Leone', '2002.2003', 'NaN', 'corruption'], 
    ['Sierra Leone', '2004.2005', 'NaN', 'corruption'], 
    ['Sierra Leone', '2006.2007', 'NaN', 'corruption'], 
    ['Sierra Leone', '2008.2009', 'NaN', 'corruption'], 
    ['Sierra Leone', '2010.2011', 'NaN', 'corruption'], 
    ['Sierra Leone', '2012.2013', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption'],
    ['Sierra Leone', '2014.2015', '87 to 98', 'corruption']
]

# Create an instance of the EmbeddingLayer and preprocess the features
embedding_layer = EmbeddingLayer(matrix_values=matrix_values)
lstm_input = embedding_layer.preprocess_features()

test_data = EmbeddingLayer(matrix_values=test_values)
input_test_data = test_data.preprocess_features()

# Example target words
target_words = ['Sierra Leone', 'has', 'a', 'value', 'of', 'NaN', 'in', 'chart', 'titled', 'corruption']

# Encode target words to numerical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(target_words)
y_train_categorical = to_categorical(y_train_encoded)

# Print shapes to verify they have the same number of samples
print(f"x_train shape: {lstm_input.shape}")  # Should print (batch_size, timesteps, features)
print(f"y_train_categorical shape: {y_train_encoded.shape}")  # Should print (batch_size, num_classes)


# Define the LSTM model
input_layer = Input(shape=(16, 4))
lstm_out = LSTM(50, return_sequences=True)(input_layer)
attention = Attention()([lstm_out, lstm_out])
lstm_out = LSTM(50)(attention)
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(lstm_out)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the shape of the input to verify
print(lstm_input.shape)  # Should print (batch_size, timesteps, features)

# Train the model
model.fit(lstm_input, y_train_categorical, epochs=500)

# Make predictions
predictions = model.predict(input_test_data)
predicted_labels = tf.argmax(predictions, axis=1)
predicted_words = label_encoder.inverse_transform(predicted_labels)

print("Predictions:")
print(predicted_words)