import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):
    def __init__(self, codes):
      super(ddiCNN, self).__init__()
      # get sizes
      n_words = codes.get_n_words()
      n_lc_words = codes.get_n_lc_words()
      n_lemmas = codes.get_n_lemmas()
      n_pos = codes.get_n_pos()
      n_labels = codes.get_n_labels()
      n_prefixes = codes.get_n_prefixes()
      n_suffixes = codes.get_n_suffixes()
      n_features = codes.get_n_feats()

      max_len = codes.maxlen

      # Embedding layers for each input type
      self.embW = nn.Embedding(n_words, 100, padding_idx=0)
      self.embLcW = nn.Embedding(n_lc_words, 100, padding_idx=0)
      self.embL = nn.Embedding(n_lemmas, 100, padding_idx=0)
      self.embP = nn.Embedding(n_pos, 100, padding_idx=0)
      self.embPr = nn.Embedding(n_prefixes, 50, padding_idx=0)
      self.embS = nn.Embedding(n_suffixes, 50, padding_idx=0)

      input_size = 100 * 4 + 50 * 2 + n_features  # 4 types of embeddings

      self.cnn = nn.Conv1d(400, 32, kernel_size=3, stride=1, padding='same')
      self.dense1 = nn.Linear(32 * max_len, 2048)
      self.linearF = nn.Linear(n_features, n_features)  # Transform features to the same dimension as suffix embeddings
      self.out = nn.Linear(32 * max_len, n_labels)

    def forward(self, w, lcw, l, p, pr, s, f):
      # Inputs should be a list: [Xw, Xlw, Xl, Xp]
      # Each embedding layer processes its corresponding input tensor
      """x = torch.cat([
         self.embW(inputs[0]),
         self.embLcW(inputs[1]),
         self.embL(inputs[2]),
         self.embP(inputs[3])
      ], dim=2)  # Concatenate along the feature dimension"""
      x = self.embW(w)
      y = self.embLcW(lcw)
      z = self.embL(l)
      a = self.embP(p)
      # b = self.embPr(pr)
      # c = self.embS(s)

      # d = f.float()
      # d = self.linearF(d)

      # Add dropouts
      # x = func.dropout(x, 0.5)
      # y = func.dropout(y, 0.5)
      # z = func.dropout(z, 0.5)
      # a = func.dropout(a, 0.5)
      # b = func.dropout(b, 0.5)
      # c = func.dropout(c, 0.5)
      # d = func.dropout(d, 0.5)

      # Concatenate embeddings
      x = torch.concat([x, y, z, a], dim=2)
      
      x = x.permute(0, 2, 1)  # Prepare for Conv1D
      x = self.cnn(x)
      x = func.relu(x)
      x = x.flatten(start_dim=1)
      return self.out(x)

