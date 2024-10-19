import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from torch.autograd import Variable
import random


class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        
        self.attention_size = attention_size
        self.projector1 = nn.Linear(2*attention_size, attention_size)
        self.projector2 = nn.Linear(attention_size, attention_size)
        self.projector3 = nn.Linear(attention_size, attention_size)
        self.projector4 = nn.Linear(attention_size, attention_size)
        self.score_layer = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outs):
        batch_sz, seq_length, features = encoder_outs.size()
        repeated_hidden = decoder_hidden.view(batch_sz, 1, features).repeat(1, seq_length, 1)
        combined_inputs = torch.cat((encoder_outs, repeated_hidden), 2).view(-1, 2*self.attention_size)

        proj1_out = self.projector1(combined_inputs)
        proj2_out = self.projector2(proj1_out)
        proj3_out = self.projector3(proj2_out)
        proj4_out = self.projector4(proj3_out)
        attn_scores = self.score_layer(proj4_out)
        attn_scores = attn_scores.view(batch_sz, seq_length)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outs).squeeze(1)
        
        return context_vector

class EncoderRNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, dropout_prob=0.3):
        super(EncoderRNN, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.gru_layer = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, features):
        batch_sz, seq_length, feature_size = features.size()    
        flattened_features = features.view(-1, feature_size)
        compressed_features = self.input_projection(flattened_features)
        dropped_features = self.dropout_layer(compressed_features)
        reshaped_features = dropped_features.view(batch_sz, seq_length, 512)

        outputs, hidden = self.gru_layer(reshaped_features)

        return outputs, hidden

class DecoderRNN(nn.Module):
    def __init__(self, decoder_hidden_size, output_size, vocab_size, word_embedding_dim=1024, dropout_rate=0.3):
        super(DecoderRNN, self).__init__()

        self.hidden_size = decoder_hidden_size
        self.output_dim = output_size
        self.vocab_size = vocab_size
        self.embedding_dim = word_embedding_dim

        self.embedding_layer = nn.Embedding(output_size, word_embedding_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.gru_layer = nn.GRU(decoder_hidden_size+word_embedding_dim, decoder_hidden_size, batch_first=True)
        self.attention_module = Attention(decoder_hidden_size)
        self.output_layer = nn.Linear(decoder_hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding_layer(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: 
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding_layer(decoder_current_input_word).squeeze(1)

            context = self.attention_module(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru_layer(gru_input, decoder_current_hidden_state)
            logprob = self.output_layer(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding_layer(decoder_current_input_word).squeeze(1)
            context = self.attention_module(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru_layer(gru_input, decoder_current_hidden_state)
            logprob = self.output_layer(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_instance, decoder_instance):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder_instance
        self.decoder = decoder_instance
    def forward(self, avi_feature, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feature)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions


