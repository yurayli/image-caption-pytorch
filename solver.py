from data_utils import *

output_path = '/output/'
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# evaluation metric
def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ')
                 if ('<end>' not in x and '<start>' not in x and '<unk>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ')
                  if ('<end>' not in x and '<start>' not in x and '<unk>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore



# solver of model with validation
class NetSolver(object):

    def __init__(self, data, model, **kwargs):
        self.data = data
        self.model = model

        # hyperparameters
        self.lr_init = kwargs.pop('lr_init', 0.001)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.step_size = kwargs.pop('step_size', 15)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.print_every = kwargs.pop('print_every', 300)
        self.checkpoint_name = kwargs.pop('checkpoint_name', 'img_caption_best.pth.tar')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_init)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.lr_decay)
        self._pad = data['word_to_idx']['<pad>']
        self._start = data['word_to_idx']['<start>']
        self._end = data['word_to_idx']['<end>']

        self.model = self.model.to(device=device)  # move the model parameters to device (CPU/GPU)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_bleu = 0.
        self.loss_history = []
        self.val_loss_history = []
        self.bleu_history = []
        self.val_bleu_history = []

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), output_path+self.checkpoint_name)
        checkpoint = {
            'optimizer': str(type(self.optimizer)),
            'scheduler': str(type(self.scheduler)),
            'lr_init': self.lr_init,
            'lr_decay': self.lr_decay,
            'step_size': self.step_size,
            'batch_size': self.batch_size,
            'epoch': epoch,
        }
        with open(output_path+'hyper_param_optim.json', 'w') as f:
            json.dump(checkpoint, f)


    def forward_net(self, features, captions):
        features = features.to(device=device)
        captions = captions.to(device=device)
        mask = (captions[:, 1:] != self._pad).view(-1)

        cap_input = captions[:, :-1]
        cap_target = captions[:, 1:]
        scores, hidden = self.model(features, cap_input)
        loss = F.cross_entropy(torch.reshape(scores, (-1, scores.size(2)))[mask],
                               torch.reshape(cap_target, (-1,))[mask],
                               size_average=False) / features.size(0)
        return loss


    def train(self, epochs):
        N = self.data['train_captions'].shape[0]

        # Start training for epochs
        for e in range(epochs):
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            self.model.train()  # set the model in "training" mode
            self.scheduler.step()  # update the scheduler
            for t, idx in enumerate(range(0, N, self.batch_size)):
                captions, features = get_batch(self.data, idx, self.batch_size)
                loss = self.forward_net(features, captions)
                if (t + 1) % self.print_every == 0:
                    bch_loss = loss.item()
                    print('t = %d, loss = %.4f' % (t+1, bch_loss))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Shuffle the data
            shuffle_data(self.data, 'train')

            # Checkpoint and record/print metrics at epoch end
            train_bleu = self.check_bleu('train', num_samples=2000)
            val_loss, val_bleu = self.check_bleu('val', num_samples=2000, check_loss=True)

            self.loss_history.append(bch_loss)
            self.val_loss_history.append(val_loss)
            self.bleu_history.append(train_bleu)
            self.val_bleu_history.append(val_bleu)

            # for floydhub metric graphs
            print('{"metric": "BLEU.", "value": %.4f}' % (train_bleu,))
            print('{"metric": "Val. BLEU.", "value": %.4f}' % (val_bleu,))
            print('{"metric": "Loss", "value": %.4f}' % (bch_loss,))
            print('{"metric": "Val. Loss", "value": %.4f}' % (val_loss,))

            if val_bleu > self.best_val_bleu:
                print('Saving model...')
                self.best_val_bleu = val_bleu
                self._save_checkpoint(e+1)
            print()


    def sample(self, features, max_length=40, b_size=3, model_mode='nic', search_mode='beam'):
        self.model.eval()  # set model to "evaluation" mode

        N = features.size(0)

        if model_mode == 'nic':

            # prepare model input
            features = features.to(device=device)
            features = self.model.dropout(self.model.bn_f(features))
            feeds = torch.ones((N, 1)) * self._start                                       # initial feed, (N, 1)
            feeds = feeds.to(device=device, dtype=torch.long)

            im_hid = self.model.dropout(self.model.relu(self.model.proj_h(features))).unsqueeze(0)
            im_state = self.model.dropout(self.model.relu(self.model.proj_c(features))).unsqueeze(0)
            h0 = torch.cat((im_hid, ) * self.model.num_layers, 0)                          # h0, c0: (L, N, H)
            #h0 = torch.zeros((self.model.num_layers, N, self.model.hidden_dim)).to(device=device)
            c0 = torch.cat((im_state, ) * self.model.num_layers, 0)
            #c0 = torch.zeros(h0.size()).to(device=device)
            hidden = (h0, c0)

            features = self.model.relu(self.model.proj_f(features)).unsqueeze(1)
            _, hidden = self.model.rnn.lstm(features, hidden)

            if search_mode == 'greedy':
                captions = self._pad * np.ones((N, max_length), dtype=np.int32)
                captions[:, 0] = np.ones(N) * self._start
                for t in range(1, max_length):
                    #word_scores, hidden = self.model.rnn(word_seq=feeds, im_feat=features,     # word_scores: (N, 1, V)
                    #                                     hidden=hidden)
                    word_scores, hidden = self.model.rnn(word_seq=feeds, hidden=hidden)
                    feeds = torch.argmax(word_scores.squeeze(1), dim=1, keepdim=True)          # feeds: (N, 1)
                    captions[:, t] = feeds.data.cpu().numpy().ravel()

            if search_mode == 'beam':
                for t in range(max_length-1):
                    scores, hidden = self.model.rnn(word_seq=feeds, hidden=hidden)
                    scores = F.softmax(scores.squeeze(), -1)

                    if t == 0:
                        scores = scores.log()   # (N, V)
                        sorted_scores, idxes = scores.topk(b_size, dim=1)   # (N, B)
                        candidates = idxes.unsqueeze(-1)   # (N, B, 1)
                        feeds = torch.reshape(candidates, (-1,1))   # (N*B, 1)

                        V = scores.size(-1)
                        h, c = hidden
                        h = torch.reshape(torch.cat([h]*b_size, -1), (h.size(0), -1, h.size(-1)))   # (L, N*B, H)
                        c = torch.reshape(torch.cat([c]*b_size, -1), (c.size(0), -1, c.size(-1)))
                        hidden = (h, c)

                    else:
                        scores = scores.log() + sorted_scores.view(-1,1)   # (N*B, V)
                        scores = torch.reshape(scores, (N, -1))   # (N, B*V)
                        sorted_scores, idxes = scores.topk(b_size, dim=1)   # (N, B)

                        prior_candidate = candidates[np.arange(N)[:,None], idxes / V]   # (N, B, t)
                        current_candidate = (idxes % V).unsqueeze(-1)   # (N, B, 1)
                        candidates = torch.cat([prior_candidate, current_candidate], -1)   # (N, B, t+1)

                        feeds = torch.reshape(candidates[:,:,-1], (-1,1))   # (N*B, 1)

                captions = candidates[:,0,:].squeeze().data.cpu().numpy()
                captions = np.concatenate([np.ones((N,1), dtype=np.int32) * self._start, captions], 1)

        if model_mode == 'attentive':

            # prepare model input
            D = features.size(1)
            features = features.to(device=device)
            features = features.transpose(1,-1)
            features = torch.reshape(features, (N, -1, D))
            feeds = torch.ones((N, 1)) * self._start                                # initial feed, (N, 1)
            feeds = feeds.to(device=device, dtype=torch.long)

            feat_mean = features.mean(1)
            h = self.model.dropout(self.model.tanh(self.model.proj_h(feat_mean)))   # (N, H)
            c = self.model.dropout(self.model.tanh(self.model.proj_c(feat_mean)))   # (N, H)

            feat_proj = self.model.proj_f(features)

            if search_mode == 'greedy':
                captions = self._pad * np.ones((N, max_length), dtype=np.int32)
                captions[:, 0] = np.ones(N) * self._start
                for t in range(1, max_length):
                    context, alpha = self.model.attn(features, feat_proj, h)   # (N, D), (N, L)
                    word_scores, h, c = self.model.rnn(feeds, context, (h,c))
                    feeds = torch.argmax(word_scores, dim=1, keepdim=True)   # feeds: (N, 1)
                    captions[:, t] = feeds.data.cpu().numpy().ravel()

            if search_mode == 'beam':
                for t in range(max_length-1):
                    if t == 0:
                        print(t)
                        context, alpha = self.model.attn(features, feat_proj, h)   # (N, D), (N, L)
                        scores, h, c = self.model.rnn(feeds, context, (h,c))
                        scores = F.softmax(scores, -1)

                        scores = scores.log()   # (N, V)
                        sorted_scores, idxes = scores.topk(b_size, dim=1)   # (N, B)
                        candidates = idxes.unsqueeze(-1).cpu()   # (N, B, 1)
                        feeds = torch.reshape(candidates, (-1,1)).cuda()   # (N*B, 1)

                        V = scores.size(-1)
                        h = torch.cat([h]*b_size, 0)   # (N*B, H)
                        c = torch.cat([c]*b_size, 0)
                        features = torch.cat([features]*b_size, 0)
                        feat_proj = torch.cat([feat_proj]*b_size, 0)

                    else:
                        print(t)
                        context, alpha = self.model.attn(features, feat_proj, h)   # (N*B, D), (N*B, L)
                        scores, h, c = self.model.rnn(feeds, context, (h,c))
                        scores = F.softmax(scores, -1)

                        scores = scores.log() + sorted_scores.view(-1,1)   # (N*B, V)
                        scores = torch.reshape(scores, (N, -1))   # (N, B*V)
                        sorted_scores, idxes = scores.topk(b_size, dim=1)   # (N, B)

                        prior_candidate = candidates[np.arange(N)[:,None], idxes / V]   # (N, B, t)
                        current_candidate = (idxes % V).unsqueeze(-1).cpu()   # (N, B, 1)
                        candidates = torch.cat([prior_candidate, current_candidate], -1)   # (N, B, t+1)

                        feeds = torch.reshape(candidates[:,:,-1], (-1,1)).cuda()   # (N*B, 1)

                captions = candidates[:,0,:].squeeze().data.cpu().numpy()
                captions = np.concatenate([np.ones((N,1), dtype=np.int32) * self._start, captions], 1)

        return captions


    def check_bleu(self, split, num_samples, batch_size=32, check_loss=False):
        captions, features, _ = sample_batch(self.data, num_samples, split)
        gt_captions = decode_captions(captions.numpy(), self.data['idx_to_word'])
        num_batches = num_samples // batch_size
        if num_samples % batch_size != 0:
            num_batches += 1

        if check_loss:
            self.model.eval()
            loss = 0
            for i in range(num_batches):
                print(i)
                start = i * batch_size
                end = (i + 1) * batch_size
                feat_batch = features[start:end].clone()
                cap_batch = captions[start:end].clone()
                loss += self.forward_net(feat_batch, cap_batch)
            loss /= num_batches

        total_score = 0.
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            sample_captions = self.sample(features[start:end])
            sample_captions = decode_captions(sample_captions, self.data['idx_to_word'])
            for gt_caption, sample_caption in zip(gt_captions[start:end], sample_captions):
                total_score += BLEU_score(gt_caption, sample_caption)

        if check_loss:
            return loss.item(), total_score / num_samples

        return total_score / num_samples

