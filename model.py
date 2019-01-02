from data_utils import *

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Decoder_A(nn.Module):
    ## Using LSTM with context input in every time step

    def __init__(self, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers):
        super(Decoder_A, self).__init__()
        # Record the arguments
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Init different layers
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(context_dim+embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, np.sqrt(1./vocab_size))
            elif isinstance(m, nn.Linear):
                truncated_normal_(m.weight, 0, np.sqrt(2./m.in_features))

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                truncated_normal_(param, 0, np.sqrt(2./param.size(1)))

    def forward(self, word_seq, im_feat, hidden):
        _, T = word_seq.size()
        emb = self.embed(word_seq)
        emb = self.dropout_emb(emb)
        im_feat = torch.cat([im_feat.unsqueeze(1)]*T, 1)   # reproduce T times to (N,T,D)
        dec_in = torch.cat([emb, im_feat], -1)
        output, hidden = self.lstm(dec_in, hidden)
        scores = self.linear(output)
        return scores, hidden


class Decoder_B(nn.Module):
    ## Using LSTM with context input only in the beginning

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
        super(Decoder_B, self).__init__()
        # Record the arguments
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Init different layers
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, np.sqrt(1./vocab_size))
            elif isinstance(m, nn.Linear):
                truncated_normal_(m.weight, 0, np.sqrt(2./m.in_features))

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                truncated_normal_(param, 0, np.sqrt(2./param.size(1)))

    def forward(self, word_seq, hidden=None, im_feat=None):
        emb = self.embed(word_seq)
        emb = self.dropout_emb(emb)
        if im_feat is None:
            output, hidden = self.lstm(emb, hidden)
            scores = self.linear(output)
            return scores, hidden
        else:
            dec_in = torch.cat([im_feat.unsqueeze(1), emb], 1)
            output, hidden = self.lstm(dec_in, hidden)
            scores = self.linear(output[:,1:,:])
            return scores, hidden


class CaptionModel(nn.Module):
    """
    Captioning Model
    context input in every time step of the decoder

    Inputs:
    - image features: Array of shape (N, D).
    - feed words: Array of shape (N, T).

    Outputs:
    - scores of words: Array of shape (N, T, V).
    - hn: Array of shape (1, N, D).
    - cn: Array of shape (1, N, D).
    """

    def __init__(self, input_dim, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionModel, self).__init__()
        # Record the arguments
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Init different layers
        self.bn_f = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.proj_f = nn.Linear(input_dim, context_dim)
        self.proj_h = nn.Linear(input_dim, hidden_dim)
        self.proj_c = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = Decoder_A(context_dim, embedding_dim, hidden_dim, vocab_size, num_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(2./m.in_features))

    def forward(self, im_feat, word_seq):
        N = im_feat.size(0)
        im_feat = self.bn_f(im_feat)
        im_hid = self.relu(self.proj_h(im_feat)).unsqueeze(0)
        im_state = self.relu(self.proj_c(im_feat)).unsqueeze(0)
        h0 = torch.cat((im_hid, ) * self.num_layers, 0)
        #h0 = torch.zeros((self.num_layers, N, self.hidden_dim)).to(device=device)
        c0 = torch.cat((im_state, ) * self.num_layers, 0)
        #c0 = torch.zeros(h0.size()).to(device=device)
        im_feat = self.relu(self.proj_f(im_feat))
        scores, hidden = self.rnn(word_seq=word_seq, im_feat=im_feat, hidden=(h0, c0))
        return scores, hidden


class CaptionModel_B(nn.Module):
    """
    Captioning Model
    context input only in the beginning of the decoder

    Inputs:
    - image features: Array of shape (N, D).
    - feed words: Array of shape (N, T).

    Outputs:
    - scores of words: Array of shape (N, T, V).
    - hn: Array of shape (1, N, H).
    - cn: Array of shape (1, N, H).
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CaptionModel_B, self).__init__()
        # Record the arguments
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Init different layers
        self.bn_f = nn.BatchNorm1d(input_dim, momentum=0.01)
        #self.proj_f = nn.Linear(input_dim, embedding_dim)
        self.proj_h = nn.Linear(input_dim, hidden_dim)
        self.proj_c = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = Decoder_B(embedding_dim, hidden_dim, vocab_size, num_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(2./m.in_features))

    def forward(self, im_feat, word_seq):
        N = im_feat.size(0)
        im_feat = self.dropout(self.bn_f(im_feat))
        im_hid = self.dropout(self.relu(self.proj_h(im_feat))).unsqueeze(0)
        im_state = self.dropout(self.relu(self.proj_c(im_feat))).unsqueeze(0)
        h0 = torch.cat((im_hid, ) * self.num_layers, 0)
        #h0 = torch.zeros((self.num_layers, N, self.hidden_dim)).to(device=device)
        c0 = torch.cat((im_state, ) * self.num_layers, 0)
        #c0 = torch.zeros(h0.size()).to(device=device)
        #im_feat = self.dropout(self.relu(self.proj_f(im_feat)))
        scores, hidden = self.rnn(word_seq=word_seq, hidden=(h0, c0))
        return scores, hidden


class Attention_layer(nn.Module):
    ## attention mechanism

    def __init__(self, proj_dim, hidden_dim):
        super(Attention_layer, self).__init__()
        # Record the arguments
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim

        # Init different layers
        self.hidden_to_attn = nn.Linear(hidden_dim, proj_dim)
        self.mid_to_alpha = nn.Linear(proj_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                truncated_normal_(m.weight, 0, np.sqrt(2./m.in_features))

    def forward(self, im_feat, feat_proj, h):
        h_att = F.relu(self.hidden_to_attn(h).unsqueeze(1) + feat_proj)   # (N, L, D')
        out_att = self.mid_to_alpha(h_att).squeeze()                      # (N, L)
        alpha = F.softmax(out_att, dim=-1)                                # (N, L)
        context = torch.sum(alpha.unsqueeze(2) * im_feat, 1)              # (N, D)
        return context, alpha


class Decoder(nn.Module):
    ## Using LSTM_cell with layer normalization

    def __init__(self, context_dim, embedding_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()
        # Record the arguments
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Init different layers
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.1)
        self.lstm_cell = nn.LSTMCell(context_dim+embedding_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, np.sqrt(1./vocab_size))
            elif isinstance(m, nn.Linear):
                truncated_normal_(m.weight, 0, np.sqrt(2./m.in_features))

        for name, param in self.lstm_cell.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                truncated_normal_(param, 0, np.sqrt(2./param.size(1)))

    def forward(self, word, context, hidden):
        emb = self.embed(word)                                           # (N, 1, M)
        emb = self.dropout_emb(emb).squeeze()
        dec_in = torch.cat([emb, context], -1)                           # (N, D+M)
        h_in, c_in = hidden
        h_in, c_in = self.ln(h_in), self.ln(c_in)
        h_output, c_output = self.lstm_cell(dec_in, (h_in, c_in))        # (N, H)
        score = self.linear(h_output)                                    # (N, V)
        return score, h_output, c_output


class AttentionModel(nn.Module):
    """
    Attentive Captioning Model

    Inputs:
    - image features: Array of shape (N, h, w, D).
    - feed words: Array of shape (N, T).

    Outputs:
    - scores of words: Array of shape (N, T, V).
    - hidden: contains (hn, cn)
        hn: Array of shape (N, H).
        cn: Array of shape (N, H).
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, vocab_size):
        super(AttentionModel, self).__init__()
        # Record the arguments
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Init different layers
        self.bn_f = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.proj_f = nn.Linear(input_dim, input_dim)
        self.proj_h = nn.Linear(input_dim, hidden_dim)
        self.proj_c = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.attn = Attention_layer(input_dim, hidden_dim)
        self.rnn = Decoder(input_dim, embedding_dim, hidden_dim, vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(2./m.in_features))

    def forward(self, im_feat, word_seq):
        N, T = word_seq.size()
        D = im_feat.size(1)
        im_feat = im_feat.transpose(1,-1)
        im_feat = torch.reshape(im_feat, (N, -1, D))                     # (N, L, D)

        # initial h, c
        feat_mean = im_feat.mean(1)
        h = self.dropout(self.tanh(self.proj_h(feat_mean)))              # (N, H)
        c = self.dropout(self.tanh(self.proj_c(feat_mean)))              # (N, H)
        #h = torch.zeros((N, self.hidden_dim)).to(device=device)
        #c = torch.zeros(h.size()).to(device=device)

        # project features
        feat_proj = self.proj_f(im_feat)

        # decoding
        scores = []
        for t in range(T):
            context, alpha = self.attn(im_feat, feat_proj, h)            # (N, D), (N, L)
            score, h, c = self.rnn(word_seq[:,t].unsqueeze(1), context, (h,c))
            scores.append(score)

        hidden = (h, c)
        scores = torch.stack(scores).transpose(1,0)                      # (N, T, V)
        return scores, hidden


def caption_model_test(feature_shape, caption_len):
    x = torch.zeros((64,)+feature_shape, dtype=dtype)
    y = torch.zeros((64,)+(caption_len,), dtype=torch.long)
    x = x.to(device=device)
    y = y.to(device=device)

    model = CaptionModel(2048, 512, 50, 100, 1000)
    model = model.to(device=device)
    scores, hidden = model(x, y[:,:-1])
    print(scores.size())
    print(hidden[0].size())
    print(hidden[1].size())


def attention_model_test():
    x = torch.zeros((64, 512, 7, 7), dtype=dtype)
    y = torch.zeros((64, 17), dtype=torch.long)
    x = x.to(device=device)
    y = y.to(device=device)

    model = AttentionModel(512, 50, 100, 1000)
    model = model.to(device=device)
    scores, hidden = model(x, y[:,:-1])
    print(scores.size())
    print(hidden[0].size())
    print(hidden[1].size())


