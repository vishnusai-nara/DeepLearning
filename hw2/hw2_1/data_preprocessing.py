import os
import re
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

def preprocess_data(filepath):
    """Preprocesses the JSON data to build word-to-index mappings and filter infrequent words."""
    with open(filepath, 'r') as f:
        file = json.load(f)

    word_counter = {}
    for entry in file:
        for caption in entry['caption']:
            words = re.sub(r'[.!,;?]', ' ', caption).split()
            for word in words:
                word = word.replace('.', '') if '.' in word else word
                word_counter[word] = word_counter.get(word, 0) + 1

    filtered_words = {word: count for word, count in word_counter.items() if count > 3}

    useful_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    index_to_word = {i: token for i, token in enumerate(useful_tokens)}
    index_to_word.update({i + len(useful_tokens): w for i, w in enumerate(filtered_words)})

    word_to_index = {w: i for i, w in index_to_word.items()}
    
    print(f"Number of filtered words: {len(filtered_words)}")
    return index_to_word, word_to_index, filtered_words


def create_minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)

    avi_data = torch.stack(avi_data, dim=0)
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        if isinstance(cap, torch.Tensor):
            targets[i, :end] = cap.clone().detach()
        else:
            targets[i, :end] = torch.tensor(cap, dtype=torch.long)

    return avi_data, targets, lengths


def annotate_captions(label_file, filtered_words, word_to_index):
    """Annotates captions by replacing words with their corresponding indices."""
    with open(label_file, 'r') as f:
        labels = json.load(f)

    annotated_captions = []
    for entry in labels:
        for caption in entry['caption']:
            sentence = [
                word_to_index.get(word, word_to_index['<UNK>']) 
                for word in re.sub(r'[.!,;?]', ' ', caption).split()
            ]
            sentence = [word_to_index['<BOS>']] + sentence + [word_to_index['<EOS>']]
            annotated_captions.append((entry['id'], sentence))
    return annotated_captions


class DatasetWithFeatures(Dataset):
    """Dataset class for loading AVI features and corresponding annotated captions."""
    def __init__(self, data_path, label_file, filtered_words, word_to_index):
        self.data_path = data_path
        self.filtered_words = filtered_words
        self.word_to_index = word_to_index
        self.avi_features = self._load_avi_features()
        self.data_pairs = annotate_captions(label_file, filtered_words, word_to_index)


    def print_insights(self, idx=None):
        """Prints dataset insights including average caption length and a sample caption."""
        unique_videos = {pair[0] for pair in self.data_pairs}
        print(f"Number of unique videos: {len(unique_videos)}")

        caption_lengths = [len(pair[1]) for pair in self.data_pairs]
        print(f"Average caption length: {np.mean(caption_lengths):.2f}")
        print(f"Caption length distribution: min={min(caption_lengths)}, "
              f"max={max(caption_lengths)}, median={np.median(caption_lengths)}")

        if idx is None:
            idx = np.random.randint(0, len(self.data_pairs))

        video_id, caption_indices = self.data_pairs[idx]
        index_to_word = {i: w for w, i in self.word_to_index.items()}
        caption_words = [index_to_word.get(idx, '<UNK>') for idx in caption_indices]
        filtered_caption = ' '.join(w for w in caption_words if w not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>'])
        
        print(f"Video ID: {video_id} and its caption: {filtered_caption}")
        
    def _load_avi_features(self):
        """Loads AVI features from .npy files."""
        return {
            os.path.splitext(file)[0]: np.load(os.path.join(self.data_path, file))
            for file in os.listdir(self.data_path)
        }

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        avi_file_name, sentence = self.data_pairs[idx]
        feature_data = torch.tensor(self.avi_features[avi_file_name], dtype=torch.float32)
        feature_data += torch.rand_like(feature_data) / 10000
        return feature_data, torch.tensor(sentence, dtype=torch.long)

class TestDataset(Dataset):
    """Dataset class for loading test AVI features."""
    def __init__(self, test_data_path):
        self.avi_features = [
            (os.path.splitext(file)[0], np.load(os.path.join(test_data_path, file)))
            for file in os.listdir(test_data_path)
        ]

    def __len__(self):
        return len(self.avi_features)

    def __getitem__(self, idx):
        return self.avi_features[idx]
