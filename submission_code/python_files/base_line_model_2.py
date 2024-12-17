# -*- coding: utf-8 -*-
"""
# Note to the grader:
As mentioned in our report, this code was sourced from a publicly available GitHub repository for research purposes. The repository can be accesed via https://github.com/RishaRane/Sarcasm_Detection_using_BERT. The code was appropriately updated and adapted to process our dataset for comparative analysis. To ensure a scientifically sound comparison, we also adjusted relevant hyperparameters, such as the number of LSTMs and training epochs.

We want to clarify that there was no intention to take credit for someone else's work.
"""



"""
# Sarcasm Detection using Hierarchical Bert

## 1. DATA LOADING AND PREPROCESSING
"""

#importing libraries
import zipfile
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

training = pd.read_csv("/content/drive/MyDrive/FALL2024/comp550/final_project/sarcasm/reddit/sarcasm_detection_reddit_training.csv")
testing = pd.read_csv("/content/drive/MyDrive/FALL2024/comp550/final_project/sarcasm/reddit/sarcasm_detection_reddit_testing.csv")
df= pd.concat([training, testing], axis=0)
df['label'] = df['label'].map({'SARCASM': 1, 'NOT_SARCASM': 0})
df = df[['response', 'label']]
df.head()

#removing unwanted numerals and symbols
df['response'] = df['response'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

#converting the data into lowercase
def lowercase(text):
  return text.lower()

df['response'] = df['response'].apply(lowercase)

"""## 2. TOKENIZATION"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#function for tokenization
def tokenize_data(text, max_length = 100):
  return tokenizer(
      text.tolist(),
      max_length = max_length,
      truncation = True,
      padding = 'max_length',
      return_tensors = 'np'
  )

tokenized_data = tokenize_data(df['response'])

tokenized_data

"""## 3. TRAIN_TEST_SPLIT"""

X = tokenized_data['input_ids']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.shape, X_test.shape

"""## 4.Building the model according to the proposed architecture"""

class HierarchicalBERT(tf.keras.Model):
    def __init__(self, bert_model, lstm_units, cnn_filters, dense_units):
        super(HierarchicalBERT, self).__init__()
        self.bert = bert_model

        # Sentence Encoding Layer
        self.dense_sentence = tf.keras.layers.Dense(768, activation='relu')

        # Context Summarization Layer
        self.mean_pooling = tf.keras.layers.GlobalAveragePooling1D()

        # Context Encoder Layer
        self.bilstm_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))

        # CNN Layer
        self.conv = tf.keras.layers.Conv1D(cnn_filters, 2, activation='relu')
        self.pool = tf.keras.layers.GlobalMaxPooling1D()

        # Fully Connected Layer
        self.dense_fc = tf.keras.layers.Dense(dense_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # BERT Embeddings
        bert_output = self.bert(inputs)[0]  # (batch_size, seq_len, hidden_size)

        # Sentence Encoding Layer
        sentence_encoded = self.dense_sentence(bert_output)  # (batch_size, seq_len, 768)

        # Context Summarization Layer
        context_summarized = self.mean_pooling(sentence_encoded)  # (batch_size, 768)

        # Expand dimensions to match the input shape required by LSTM
        context_summarized = tf.expand_dims(context_summarized, 1)  # (batch_size, 1, 768)

        # Context Encoder Layer
        context_encoded = self.bilstm_encoder(context_summarized)  # (batch_size, 1, 2 * lstm_units)

        # Squeeze the second dimension to match the input shape required by Conv1D
        context_encoded_squeezed = tf.squeeze(context_encoded, axis=1)  # (batch_size, 2 * lstm_units)

        # Add the channels dimension to match the input shape required by Conv1D
        context_encoded_expanded = tf.expand_dims(context_encoded_squeezed, axis=-1)  # (batch_size, 2 * lstm_units, 1)

        # CNN Layer
        conv_output = self.conv(context_encoded_expanded)  # (batch_size, new_seq_len, cnn_filters)
        pooled_output = self.pool(conv_output)  # (batch_size, cnn_filters)

        # Fully Connected Layer
        dense_output = self.dense_fc(pooled_output)  # (batch_size, dense_units)

        # Output Layer
        final_output = self.output_layer(dense_output)  # (batch_size, 1)
        return final_output

#loading the pretrained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

#defining the hierarchical bert model
model = HierarchicalBERT(bert_model, lstm_units = 300, cnn_filters = 64, dense_units = 32)

model.compile(optimizer = 'adamW',loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 25, batch_size = 32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy : {accuracy * 100}')