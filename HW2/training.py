import sys
import time
import torch
import pickle
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_preprocessing import DatasetWithFeatures, create_minibatch, preprocess_data
from models import Seq2SeqModel, EncoderRNN, DecoderRNN

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, epoch, loss_fn, optimizer, train_loader):
    """Trains the model for one epoch."""
    model.train()  # Set model to training mode
    start_time = time.time()
    epoch_loss = 0

    for batch in train_loader:
        avi_features, ground_truths, lengths = batch

        # Move data to GPU
        avi_features = avi_features.to(device, non_blocking=True)
        ground_truths = ground_truths.to(device, non_blocking=True)

        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        seq_logProb, _ = model(avi_features, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        # Ignore <BOS> token for loss calculation
        ground_truths = ground_truths[:, 1:]

        # Calculate loss
        loss = calculate_loss(seq_logProb, ground_truths, lengths, loss_fn) / len(seq_logProb)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        epoch_loss += loss.item()

    elapsed_time = time.time() - start_time
    avg_epoch_loss = epoch_loss / len(train_loader)

    print(f"Epoch: {epoch}, Loss: {avg_epoch_loss:.4f}")

    return epoch_loss, elapsed_time

def calculate_loss(seq_logProb, ground_truths, lengths, loss_fn):
    """Calculates the loss for a batch."""
    batch_size = len(seq_logProb)

    # Concatenate predictions and ground truths by length
    concatenated_predictions = torch.cat(
        [seq_logProb[i][:lengths[i] - 1] for i in range(batch_size)], dim=0
    )
    concatenated_ground_truths = torch.cat(
        [ground_truths[i][:lengths[i] - 1] for i in range(batch_size)], dim=0
    )

    # Calculate loss
    return loss_fn(concatenated_predictions, concatenated_ground_truths)

def test_model(test_loader, model, index_to_word, beam_size=5):
    """Evaluates the model and returns predictions."""
    model.eval()
    results = []

    with torch.no_grad():
        for video_ids, avi_features in test_loader:
            avi_features = avi_features.cuda().float()
            _, predictions = model(avi_features, mode='inference')

            formatted_results = format_predictions(predictions, index_to_word)
            results.extend(zip(video_ids, formatted_results))
    
    return results

def format_predictions(predictions, index_to_word):
    """Formats the model predictions into readable captions."""
    return [
        ' '.join([index_to_word[idx.item()] if index_to_word[idx.item()] != '<UNK>' else 'something' for idx in s]).split('<EOS>')[0]
        for s in predictions
    ]

def main():
    """Main function to run the training process."""
    print("Specify the training data features path as the first argument and 'training_label.json' as the second.")
    
    files_dir, label_file = sys.argv[1], sys.argv[2]

    # Preprocess data and move model to GPU
    index_to_word, word_to_index, filtered_words = preprocess_data(label_file)
    encoder = EncoderRNN().to(device)  # Move encoder to GPU
    decoder = DecoderRNN(512, len(index_to_word) + 4, len(index_to_word) + 4, 1024, 0.3).to(device)
    model = Seq2SeqModel(encoder, decoder).to(device)  # Move Seq2Seq model to GPU

    # Load dataset and DataLoader
    train_dataset = DatasetWithFeatures(files_dir, label_file, filtered_words, word_to_index)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=create_minibatch)

    # Print insights about the dataset
    train_dataset.print_insights(idx=10)

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for the specified number of epochs
    losses, total_training_time = train(model, loss_fn, optimizer, train_loader, epochs=200)

    # Save the trained model
    torch.save(model.state_dict(), "model_vishnu.h5")
    print(f"Training finished. Total training time: {total_training_time:.2f} seconds")

    # Save the losses to a file
    with open('losses.txt', 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")

def train(model, loss_fn, optimizer, train_loader, epochs):
    """Trains the model for the given number of epochs."""
    losses = []
    total_time = 0

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_time = train_model(model, epoch, loss_fn, optimizer, train_loader)
        losses.append(epoch_loss)
        total_time += epoch_time

    return losses, total_time


def save_losses(losses, filename):
    """Saves the training losses to a file."""
    with open(filename, 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")

if __name__ == "__main__":
    main()



