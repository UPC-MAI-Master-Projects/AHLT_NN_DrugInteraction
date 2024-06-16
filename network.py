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
        self.embW = nn.Embedding(n_words, 100)
        self.embLcW = nn.Embedding(n_lc_words, 100)
        self.embL = nn.Embedding(n_lemmas, 100)
        self.embP = nn.Embedding(n_pos, 100)
        self.embPr = nn.Embedding(n_prefixes, 50)
        self.embS = nn.Embedding(n_suffixes, 50)

        # Calculate the total input size for LSTM
        input_size = 100 * 4 + 50 * 2 + n_features

        # LSTM layer
        self.lstm = nn.LSTM(input_size, 400, num_layers=2, batch_first=True, bidirectional=True)

        # Convolutional layer
        self.cnn = nn.Conv1d(800, 32, kernel_size=3, stride=1, padding='same')  # Note: 800 due to bidirectional LSTM

        # Dense layers
        self.head1 = nn.Linear(32 * max_len, 256)  # Adjusting to the output of CNN layer
        self.linearF = nn.Linear(n_features, n_features)  # For processing additional features

        # Output layer
        self.out = nn.Linear(256, n_labels)  # Assuming a flattening or global pooling before this layer

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, w, lcw, l, p, pr, s, f):
        # Embeddings
        emb_w = self.dropout(self.embW(w))
        emb_lcw = self.dropout(self.embLcW(lcw))
        emb_l = self.dropout(self.embL(l))
        emb_p = self.dropout(self.embP(p))
        emb_pr = self.dropout(self.embPr(pr))
        emb_s = self.dropout(self.embS(s))

        # Convert additional features to float and process
        z = self.linearF(f.float())

        # Concatenate all embeddings and features
        x = torch.cat([emb_w, emb_lcw, emb_l, emb_p, emb_pr, emb_s, z], dim=2)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Prepare for Conv1D
        lstm_out = lstm_out.permute(0, 2, 1)  # Rearrange to (batch, channels, length)

        # CNN processing
        cnn_out = self.cnn(lstm_out)
        cnn_out = func.relu(cnn_out)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten CNN output

        # Pass through dense layer
        dense_out = self.head1(cnn_out)
        dense_out = func.relu(dense_out)

        # Output layer
        output = self.out(dense_out)
        return output
