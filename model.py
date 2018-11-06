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


class Decoder(nn.Module):
    ## Language Model using LSTM

    def __init__(self, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers):
        super(Decoder, self).__init__()
        # Record the arguments
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Init different layers
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.2)
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
        im_feat = torch.cat([im_feat.unsqueeze(1)]*T, 1)
        emb = torch.cat([emb, im_feat], -1)
        output, hidden = self.lstm(emb, hidden)
        scores = self.linear(output)
        return scores, hidden


class CaptionModel(nn.Module):
    """
    Captioning Model

    Inputs:
    - init hidden features: Array of shape (N, D).
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
        self.rnn = Decoder(context_dim, embedding_dim, hidden_dim, vocab_size, num_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(2./m.in_features))

    def forward(self, im_feat, word_seq):
        N = im_feat.size(0)
        im_feat = F.dropout(self.bn_f(im_feat), 0.2, training=self.training)
        im_hid = self.dropout(self.relu(self.proj_h(im_feat))).unsqueeze(0)
        im_state = self.dropout(self.relu(self.proj_c(im_feat))).unsqueeze(0)
        h0 = torch.cat((im_hid, ) * self.num_layers, 0)
        #h0 = torch.zeros((self.num_layers, N, self.hidden_dim)).to(device=device)
        c0 = torch.cat((im_state, ) * self.num_layers, 0)
        #c0 = torch.zeros(h0.size()).to(device=device)
        im_feat = self.dropout(self.relu(self.proj_f(im_feat)))
        scores, hidden = self.rnn(word_seq=word_seq, im_feat=im_feat, hidden=(h0, c0))
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

