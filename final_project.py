import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertModel, BertModel
from sklearn.model_selection import train_test_split
import json
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import ast

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Default to CUDA
else:
    torch.set_default_tensor_type(torch.FloatTensor)  # Default to CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(device)


def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def list_shape(lst):
    """
    Recursively determines the shape of a nested list.

    Args:
        lst (list): The input nested list.

    Returns:
        tuple: A tuple representing the shape of the list.
    """
    if isinstance(lst, list):
        if len(lst) == 0:
            return (0,)  # Empty list
        return (len(lst), *list_shape(lst[0]))
    else:
        return ()  # End of recursion for non-list elements


def custom_collate_fn(batch):
    contexts = [sample['context'] for sample in batch]
    responses = [sample['response'] for sample in batch]
    labels = torch.stack([sample['label'] for sample in batch])

    # Process contexts and responses as needed
    # For example, flattening or stacking them appropriately
    return {
        'context': contexts,
        'response': responses,
        'label': labels
    }

"""# Data Visualization

## Path Constants

## JSON files format (only use once to convert it to dataframe)
"""

REDDIT_TRAINING_DATA_PATH_JSON = "sarcasm/reddit/sarcasm_detection_shared_task_reddit_training.jsonl"
REDDIT_TESTING_DATA_PATH_JSON = "sarcasm/reddit/sarcasm_detection_shared_task_reddit_testing.jsonl"
TWITTER_TRAINING_DATA_PATH_JSON = "sarcasm/twitter/sarcasm_detection_shared_task_twitter_training.jsonl"
TWITTER_TESTING_DATA_PATH_JSON= "sarcasm/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl"

"""## Data frame format"""

REDDIT_TRAINING_PATH_DATAFRAME = "sarcasm_detection_reddit_training.csv"
REDDIT_TESTING_PATH_DATAFRAME = "sarcasm_detection_reddit_testing.csv"

"""# Data Processing

"""

def is_pad_tensor(tensor):
    correct_start = tensor[0][0] == 101
    correct_middle = tensor[0][1] == 0
    correct_end = tensor[0][2] == 102
    return correct_start and correct_middle and correct_end


class RedditDataset(Dataset):
    def __init__(self, path_to_dataset_df, device,  a_sample_of = None, tokenizer = "bert-base-cased" , max_sentence_length=100):
        # Load data from json file to dataframe
        if path_to_dataset_df.endswith('.jsonl'):
            print("Recevied a json file")
            self.data = pd.DataFrame(load_jsonl_data(path_to_dataset_df))
        else:
            print("Recevied a csv file")
            self.data = pd.read_csv(path_to_dataset_df)
            self.data['context']= self.data['context'].apply(ast.literal_eval)


        # Added this feature to use it if we want to check if the code trains well on a subset
        if a_sample_of is not None:
            self.data = self.data.sample(n=a_sample_of, random_state=42)

        # Map string labels to floats
        self.data['label'] = self.data['label'].map({'SARCASM': 1, 'NOT_SARCASM': 0}).astype(float)

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] # Here we obtain the row that contain the context, response and label. All readable, nothing tokenized yet

        context = row['context'] # Untokenized context

        context_tokens = []
        for utterence in context: # Here we are going to tokenize each utterencein the context
            tokenized_utterence = self.tokenizer(
                utterence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_sentence_length
            ).to(self.device)

            context_tokens.append(tokenized_utterence)


        response = row['response']
        response_tokens = self.tokenizer(
            response,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sentence_length
        ).to(self.device)

        label = torch.tensor(float(row['label']), dtype=torch.float32).to(self.device)
        return {'context': context_tokens, 'response': response_tokens, 'label': label}

    def get_df(self):
        return self.data

"""# Model Architecture as per the article

I will build the model architure exactly as shown in the article

## Layer 1: Sentence Encoding Layer
"""

# Input: batches of Context (utterences), and response

# output: two outputs:
    # batch of Context Encoded (batch_size, m, 100, 768)
    # batch of Reponse Encoded list of size (batch_size, 100, 2*lstm_hiddien_size)
class SentenceEncodingLayer(nn.Module):

    # The lstm_hidden_size is a parameter that we can adjust
    def __init__(self, bert_model_name="bert-base-cased", lstm_hidden_size=300):

        super(SentenceEncodingLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT tokenizer and model for context and response
        self.context_encoder = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.response_encoder = BertModel.from_pretrained(bert_model_name).to(self.device)

        # BiLSTM layer for response encoding
        self.lstm = nn.LSTM(
            input_size=768,  # BERT output dimension
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        ).to(self.device)


    def encode_context(self, context_tokens_batch): # Outputs are stacked to form a tensor of shape (m, max_sentence_length, 768).
        batched_context_embeddings = []


        for tokenized_context in context_tokens_batch:

            utterance_embeddings = []

            for utt in tokenized_context:
               # if is_pad_tensor(utt['input_ids']):
               #     continue
                utt_embedding = self.context_encoder(**utt).last_hidden_state
                utterance_embeddings.append(utt_embedding.squeeze(0))


            stacked_utterance_embeddings = torch.stack(utterance_embeddings)
            batched_context_embeddings.append(stacked_utterance_embeddings)

        stacked_batched_context_embeddings = torch.stack(batched_context_embeddings)
        return stacked_batched_context_embeddings

    def encode_response(self, response_tokens_batch):
        # The input is a batch of respones
        """
        Encodes the response sentence using the response BERT encoder and a BiLSTM.
        A BiLSTM processes the embeddings to capture sequential dependencies,
        outputting a representation of shape (batch_size, max_sentence_length, 2*lstm_hidden_size).
        """
        batched_response_embeddings = []
        for tokenized_response in response_tokens_batch:
            response_embedding = self.response_encoder(**tokenized_response).last_hidden_state  # Shape: (1, max_sentence_length, 768)
            lstm_out, _ = self.lstm(response_embedding)  # Shape: (1, max_sentence_length, 2*lstm_hidden_size)
            batched_response_embeddings.append(lstm_out)
        # Stack embeddings for all responses in the batch

        lstm_out = lstm_out = torch.cat(batched_response_embeddings, dim=0) #torch.stack(batched_response_embeddings)

        return lstm_out

    def forward(self, context_tokens_batch, response_tokens_batch):
        """
        Encodes the context and response and returns their representations.
        """
        # Encode the context (list of utterances)
        context_representation = self.encode_context(context_tokens_batch)

        # Encode the response
        response_representation = self.encode_response(response_tokens_batch)
        return context_representation, response_representation

"""## Layer 2: Context Summarization

This layer only operates on the "Encoded_Context" according to the article only deal with the "encoded_context" which is outputted from the previous layer. The each encoded context has a dimension of [m, 100, 768] and in total we have len(dataset) contexts.
"""

# This thing only deals with the Context (It does not care about the response)
# The goal is to reduce the dimension of the context


# input shape: [batch_size num_uttenrences, 100, 768]

# output: [batch_size, num_uttenrences - k_row + 1, 100 - k_col +1, 128 ]

class ContextSummarizationLayer(nn.Module):
    # The dsum, k_rows, k_col are variables that need to be optimized
    def __init__(self, d_bert= 768, dsum = 128, k_row = 2, k_col =2, padding = 0, stride =1):
        super(ContextSummarizationLayer, self).__init__()
        self.k_row = k_row
        self.k_col = k_col

        # Conv2D expects an input as follows: (N, C_in, H, W)
        self.conv2D = nn.Conv2d(in_channels=d_bert,  # Context tensor is treated as a single "channel"
                                out_channels=dsum, # Number of features maps to produce, which means that each utterence will be mapped to d_sum features
                                kernel_size=(k_row, k_col),
                                stride = stride,     # I think this is the default value, but just putting it to ensure the understanding flow of the article
                                padding = padding  # Does the article mentions padding? I think that's a different aspect to look after
                                )

    def forward(self, context_representation_batch):  # Input shape: (Batch_size ,m, d_sen, d_bert) which is (Batch_size , num_utterences, 100, 768)
        # Permute from (Batch_size , m, d_sen, d_bert), to (Batch_size , d_bert, m, d_sen) to match Conv2D requirements.
        context_representation_batch = context_representation_batch.permute(0, 3, 1, 2).to(self.conv2D.weight.device)
        # Apply 2D convolution the input shape here is (d_bert, m, d_sen)
        summarized_context_batch = self.conv2D(context_representation_batch) # Shape: Output shape: (Batch_size , dsum, m-k_row+1, d_sen-k_col+1).

        # Permute the dimensions to make the output compatible with what the article says: (Batch_size , dsum, m-k_row+1, d_sen-k_col+1) -> (Batch_size , m-k_row+1, d_sen-k_col+1, dsum)
        summarized_context_batch = summarized_context_batch.permute(0, 2, 3, 1)

        return summarized_context_batch

"""## Layer 3: Context Encoder Layer"""

# input shape: (batch_size, m-k_row+1, d_sen-k_col+1, dsum)
# output_shape: (batch_size, m-k_row+1 , 2*lstm_hidden_size)
class ContextEncoderLayer(nn.Module):
    # The article says that the number of layers is 1
    def __init__(self, size_of_input, lstm_hidden_size=300, number_of_layers=1):
        super(ContextEncoderLayer, self).__init__()
        self.input_size = size_of_input
        self.lstm_hidden_size = lstm_hidden_size
        self.bilstm = nn.LSTM(
            input_size=self.input_size, # This should be the dimension of the input for each utterance
            hidden_size=self.lstm_hidden_size,
            num_layers=number_of_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, summarized_context_batch):

        # Input shape of the summarized context is: (batch_size, m-k_row+1, d_sen-k_col+1, dsum)
        batch_size, reduced_num_utterences, reduce_dsen, dsum = summarized_context_batch.size()

        # The reshape operation flattens dsen - kcol + 1 and dsum into a single dimension input preserving m- krows + 1 as the sequence length.
        summarized_context_batch = summarized_context_batch.reshape(batch_size, reduced_num_utterences, -1)  # Shape: (batch_size, m - k_row + 1, (d_sen - k_col + 1) * d_sum)
        summarized_context_batch = summarized_context_batch.to(self.bilstm.weight_hh_l0.device) # This is just to make sure we are on the correct device, maybe need to remove it
        lstm_output, _ = self.bilstm(summarized_context_batch)

        # Output shape: (batch_size ,m - k_row + 1, hidden_dim * 2)
        return lstm_output

"""## Layer 4: CNN Layer

This takes both the "context" outputted from the bilstm ContextEncoderLayer, and the response that we had in the first place.

For the context, its dimensions is <M, d_lstm> where M is the reduced number of utterences after convolution, and d_lstm is the number of hidden units times two because it is biredctional

For the response, its dimension is <d_sen, d_lstm>

The article specifies combining the context representation (o_{con}) and the response representation (o_{res}). `so we need to handle this concatination BEFORE entering this layer, this layer takes as an input the concatinated stuff`
"""

class CNNLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes):
        super(CNNLayer, self).__init__()


        # Conv2D expects an input as follows: (N, C_in, H, W)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=1
            ) for kernel_size in kernel_sizes
        ])

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, combined_representation_batch):
        # combined representation: concatination(response,context)

        feature_maps = []

        for conv in self.convs:

            conv_output = torch.relu(conv(combined_representation_batch))


            pooled_output = self.maxpool(conv_output)

            feature_maps.append(pooled_output)

        # Initialize an empty list to store flattened feature maps
        flattened_maps = []

        # Loop through the feature maps and flatten them
        for feature_map in feature_maps:
            # Get the batch size and number of features
            batch_size = feature_map.size(0)  # Number of samples in the batch
            num_features = feature_map.size(1)  # Number of features per sample

            # Flatten
            flattened_map = feature_map.view(batch_size, num_features)

            # Append the flattened map to the list
            flattened_maps.append(flattened_map)

        # Concatenate all flattened feature maps along the feature dimension
        # This combines features from all convolutional layers into one vector
        unified_feature_vector = torch.cat(flattened_maps, dim=1)

        # Return the final feature vector
        return unified_feature_vector

"""## Layer 5: Fully Connected Layer"""

import torch.nn.init as init

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_features):
        super(FullyConnectedLayer, self).__init__()
        # Fully connected layer
        self.fc = nn.Linear(input_features, 1)  # Output size is 1 (binary classification)

        # Explicit initialization
        init.xavier_uniform_(self.fc.weight)  # Xavier initialization for weights
        init.zeros_(self.fc.bias)             # Initialize bias to 0

    def forward(self, x):
        # Forward pass through the fully connected layer and sigmoid activation
        x = self.fc(x) # No sigmoid here; use BCEWithLogitsLoss
        return x

"""## Full Architecutre (All Layers Combined Together)"""

class HierarchicalBERT(nn.Module):
    # Define class attributes (constants shared by all instances of the class)
    GLOBAL_D_BERT = 768
    GLOBAL_STRIDE = 1
    GLOBAL_PADDING = 0
    GLOBAL_K_ROW = 2
    GLOBAL_K_COL = 2

    def __init__(self, bert_model_name="bert-base-cased", lstm_hidden_size=300, max_sentence_length=100,
                 output_channels=128, kernel_sizes=[(2, 2), (2, 3), (2, 5)], num_classes=1):
        super(HierarchicalBERT, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentence_encoder = SentenceEncodingLayer(
            bert_model_name=bert_model_name,
            lstm_hidden_size=lstm_hidden_size,
        ).to(device)

        self.context_summarizer = ContextSummarizationLayer(
            d_bert=self.GLOBAL_D_BERT,
            dsum=output_channels,
            k_row=self.GLOBAL_K_ROW,
            k_col=self.GLOBAL_K_COL
        ).to(device)

        self.context_encoder = ContextEncoderLayer(
            size_of_input=(output_channels * (max_sentence_length - self.GLOBAL_K_ROW + 1)),
            lstm_hidden_size=lstm_hidden_size
        ).to(device)

        self.cnn_layer = CNNLayer(
            input_channels=1,
            output_channels=output_channels,
            kernel_sizes=kernel_sizes
        ).to(device)

        self.fc_layer = FullyConnectedLayer(
            input_features=len(kernel_sizes) * output_channels
        ).to(device)


    def forward(self, context, response, p=0):
        """
        Forward pass through the entire model.
        Args:
            context (List[str]): List of utterances (context).
            response (str): Response to classify as sarcastic or not.
        Returns:
            Tensor: Probability of sarcasm for the response.
        """
        if p ==1 :print("Received input")
        if p ==1 :print(f"Number of utterences: {len(context)}")
        if p ==1 :print(f"Response: {response}")
        # 1. Sentence Encoding
        context_representation, response_representation = self.sentence_encoder(context, response)
        if p ==1 :print("After SentenceEncodingLayer, Here is the input to the context Summarization Layer")
        if p ==1 :print(f"Context Representation Shape: {context_representation.shape}")
        if p ==1 :print(f"Response Representation Shape: {response_representation.shape}")

        # 2. Context Summarization
        summarized_context = self.context_summarizer(context_representation)
        if p ==1 :print(f"Summarized Context Representation Shape: {summarized_context.shape}")


        # 3. Context Encoding
        encoded_context = self.context_encoder(summarized_context)
        if p ==1 :print(f"Encoded Context Representation Shape: {encoded_context.shape}")

        # 4. Concatenate Context and Response
        if p ==1 :print("Concatenating Context and Response")
        if p ==1 :print(f"Context Representation Shape: {encoded_context.shape}")
        if p ==1 :print(f"Response Representation Shape: {response_representation.shape}")

        combined_representation = torch.cat((encoded_context, response_representation), dim=1) # I changed this from dim 0 to dim 1 since both match except at dim 1
        combined_representation = combined_representation.unsqueeze(1) # add channel dims
        if p ==1 :print(f"Combined Representation Shape: {combined_representation.shape}")
        # 5. CNN Layer
        cnn_output = self.cnn_layer(combined_representation)
        if p ==1 :print(f"CNN Output Shape: {cnn_output.shape}")

        # 6. Fully Connected Layer
        logits = self.fc_layer(cnn_output)
        if p ==1: print(f"Final Output: {logits}")

        if p ==1: print("================= \n")

        return logits

"""# Training Loop"""

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    training_epoch_loss = 0
    correct_training_predictions = 0
    total_training_predictions = 0
    progress_bar = tqdm(data_loader, leave=True, desc="Training Progress")
    for batch in progress_bar:
        # Move batch data to the device
        context = batch['context']
        response = batch['response']
        labels = batch['label'].unsqueeze(1).to(device).float()

        optimizer.zero_grad()

        # Forward pass
        logits = model(context, response)  # Remove p=0
        loss = criterion(logits, labels)
        predictions = (torch.sigmoid(logits) > 0.5).long()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update metrics
        training_epoch_loss += loss.item()
        total_training_predictions += labels.size(0)
        correct_training_predictions += (predictions == labels.long()).sum().item()
        batch_accuracy = (predictions == labels.long()).sum().item() / labels.size(0)
        progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': batch_accuracy})

    epoch_loss = training_epoch_loss / len(data_loader)
    epoch_accuracy = correct_training_predictions / total_training_predictions

    return epoch_loss, epoch_accuracy


def validation(model, val_loader, criterion, device):
    """
    Validate the model on the validation dataset and return loss and accuracy.
    """
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move data to the device
            context = batch['context']
            response = batch['response']
            labels = batch['label'].unsqueeze(1)  # Ensure correct shape

            # Forward pass
            logits = model(context, response)
            loss = criterion(logits, labels.float())

            # Compute predictions
            predictions = (torch.sigmoid(logits) > 0.5).long()

            # Update metrics
            epoch_loss += loss.item()
            total_predictions += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()


    # Calculate average loss and accuracy
    avg_loss = epoch_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    print(f"[Validation] Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%\n")
    return avg_loss, accuracy

"""## Dataset and model initialization"""

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reddit_dataset = RedditDataset(REDDIT_TRAINING_PATH_DATAFRAME, device, 64)
data_loader = DataLoader(reddit_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn, generator=torch.Generator(device=device))

"""## Loss Function"""

criterion = nn.BCEWithLogitsLoss()

"""## Optimizer"""

model = HierarchicalBERT().to(device)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#for name, param in model.named_parameters():
#    print(f"Layer: {name} | Mean: {param.mean().item()} | Std: {param.std().item()}")

"""## Training"""

from tqdm import tqdm
import torch

num_epochs = 25

epoch_train_loss = []
epoch_train_accuracy = []
epoch_val_loss = []
epoch_val_accuracy = []
progress_bar = tqdm(range(num_epochs), desc="Epochs")
for epoch in progress_bar:
    train_loss, train_accuracy = train_one_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    scheduler.step()
    epoch_train_loss.append(train_loss)
    epoch_train_accuracy.append(train_accuracy)
    print(f"[Training] Loss: {train_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%")

# Plot training loss and training accuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epoch_train_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch_train_accuracy, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

"""
    # Validation after training
    val_loss, val_accuracy = validation(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )

    # Print results for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"    [Training] Loss: {train_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%")
    print(f"    [Validation] Loss: {val_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%\n")
    """