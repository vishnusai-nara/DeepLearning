import sys
import torch
import json
import pickle
from models import EncoderRNN, DecoderRNN, Seq2SeqModel  # Ensure your model imports are correct
from training import test_model
from data_preprocessing import TestDataset
from torch.utils.data import DataLoader
from bleu_eval import BLEU

# Define device to ensure compatibility with CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, vocab_size, device=device):
    """Loads a pre-trained model from a state dictionary."""
    # Initialize model architecture (ensure it matches the saved model)
    encoder = EncoderRNN().to(device)
    decoder = DecoderRNN(512, vocab_size, vocab_size, 1024, 0.3).to(device)
    model = Seq2SeqModel(encoder, decoder).to(device)

    # Load the saved state dictionary and apply it to the model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model

def load_index_to_word(path):
    """Loads the index-to-word mapping from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def write_predictions(loader, model, index_to_word, output_path):
    """Writes model predictions to an output file."""
    model.eval()  # Ensure model is in evaluation mode
    predictions = test_model(loader, model, index_to_word)
    with open(output_path, 'w') as f:
        f.writelines(f'{video_id},{caption}\n' for video_id, caption in predictions)

def calculate_bleu_score(label_path, output_path):
    """Calculates the average BLEU score based on model predictions."""
    with open(label_path, 'r') as f:
        test_data = json.load(f)

    with open(output_path, 'r') as f:
        result = {line.split(',', 1)[0]: line.split(',', 1)[1].strip() for line in f}

    bleu_scores = [
        BLEU(result[item['id']], [cap.rstrip('.') for cap in item['caption']], True)
        for item in test_data
    ]
    return sum(bleu_scores) / len(bleu_scores)

def main(test_features_dir, output_path):
    print("Ensure the model, index-to-word pickle, and 'testing_label.json' are available.")

    # Load index-to-word mapping
    index_to_word = load_index_to_word('index2word.pickle')
    vocab_size = len(index_to_word) + 4  # Adjust based on your vocabulary size

    # Load model with correct vocabulary size
    model = load_model('model_vishnu.h5', vocab_size)

    # Load test dataset and DataLoader
    test_dataset = TestDataset(f'{test_features_dir}/feat')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Write predictions to the output file
    write_predictions(test_loader, model, index_to_word, output_path)

    # Calculate and print the average BLEU score
    avg_bleu_score = calculate_bleu_score('MLDS_hw2_1_data/testing_label.json', output_path)
    print(f"Average BLEU score: {avg_bleu_score}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <test_features_dir> <output_file>")
        sys.exit(1)

    test_features_dir, output_file = sys.argv[1], sys.argv[2]
    main(test_features_dir, output_file)
