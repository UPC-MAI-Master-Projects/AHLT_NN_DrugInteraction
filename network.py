import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):
    def __init__(self, codes):
      super(ddiCNN, self).__init__()
      # Extract sizes from the 'codes' object
      n_words = codes.get_n_words()
      n_lc_words = codes.get_n_lc_words()
      n_lemmas = codes.get_n_lemmas()
      n_pos = codes.get_n_pos()
      n_prefixes = codes.get_n_prefixes()
      n_suffixes = codes.get_n_suffixes()
      n_features = codes.get_n_feats()  # Additional features dimension
      n_labels = codes.get_n_labels()
      max_len = codes.maxlen

      # Embedding layers
      self.embW = nn.Embedding(n_words, 100, padding_idx=0)
      self.embLcW = nn.Embedding(n_lc_words, 100, padding_idx=0)
      self.embL = nn.Embedding(n_lemmas, 100, padding_idx=0)
      self.embP = nn.Embedding(n_pos, 100, padding_idx=0)
      self.embPr = nn.Embedding(n_prefixes, 50, padding_idx=0)
      self.embS = nn.Embedding(n_suffixes, 50, padding_idx=0)

      # Calculate the total input size for LSTM
      input_size = 100 * 4 + 50 * 2 + n_features  # 4 embeddings of 100 features each, 2 of 50 features

      # LSTM layer
      self.lstm = nn.LSTM(input_size, 400, num_layers=2, batch_first=True, bidirectional=True)

      # Convolutional layer
      self.cnn = nn.Conv1d(800, 32, kernel_size=3, stride=1, padding='same')  # Note: 800 due to bidirectional LSTM

      # Dense layers
      self.dense1 = nn.Linear(32 * max_len, 2048)
      self.linearF = nn.Linear(n_features, n_features)  # For processing additional features

      # Output layer
      self.out = nn.Linear(2048, n_labels)

      # Dropout layer
      self.dropout = nn.Dropout(0.5)

    def forward(self, w, lcw, l, p, pr, s, f):
      # Embeddings
      emb_w = self.embW(w)
      emb_lcw = self.embLcW(lcw)
      emb_l = self.embL(l)
      emb_p = self.embP(p)
      emb_pr = self.embPr(pr)
      emb_s = self.embS(s)

      # Feature transformation
      f_transformed = self.linearF(f.float())

      # Concatenate all embeddings and features
      combined = torch.cat([emb_w, emb_lcw, emb_l, emb_p, emb_pr, emb_s, f_transformed], dim=2)
      
      # Apply dropout to the combined embeddings
      combined = self.dropout(combined)

      # LSTM processing
      lstm_out, _ = self.lstm(combined)

      # Apply dropout to LSTM output
      lstm_out = self.dropout(lstm_out)

      # Prepare for Conv1D
      lstm_out = lstm_out.permute(0, 2, 1)  # From (N, L, C) to (N, C, L)

      # CNN processing
      cnn_out = self.cnn(lstm_out)
      cnn_out = func.relu(cnn_out)

      # Apply dropout to CNN output
      cnn_out = self.dropout(cnn_out)

      # Flatten the output for the dense layer
      cnn_out = cnn_out.flatten(start_dim=1)

      # Final dense layer before output
      dense_out = self.dense1(cnn_out)
      output = self.out(dense_out)

      return output

