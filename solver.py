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

    def train(self, epochs):
        N = self.data['train_captions'].shape[0]

        # Start training for epochs
        for e in range(epochs):
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            self.model.train()  # set the model in "training" mode
            self.scheduler.step()  # update the scheduler
            for t, idx in enumerate(range(0, N, self.batch_size)):
                captions, features = get_batch(self.data, idx, self.batch_size)
                features = features.to(device=device)
                captions = captions.to(device=device)
                mask = (captions[:, 1:] != self._pad).view(-1)

                cap_input = captions[:, :-1]
                cap_target = captions[:, 1:].contiguous()
                scores, hidden = self.model(features, cap_input)
                loss = torch.sum(F.cross_entropy(scores.view(-1, scores.size(2))[mask],
                                                 cap_target.view(-1)[mask],
                                                 reduce=False)) / self.batch_size
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
            val_loss, val_bleu = self.check_bleu('val',
                                    num_samples=2000, check_loss=True)

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

    def sample(self, features, max_length=40):
        self.model.eval()  # set model to "evaluation" mode

        N = features.size(0)
        captions = self._pad * np.ones((N, max_length), dtype=np.int32)
        is_end = np.zeros(N).astype(np.bool)  # mark if a sequence has ended, (batch_size, )

        # prepare model input
        features = features.to(device=device)
        features = F.dropout(self.model.bn_f(features), 0.2, self.model.training)
        features = self.model.dropout(features)
        feeds = torch.ones((N, 1)) * self._start                                       # initial feed, (batch_size, 1)
        feeds = feeds.to(device=device, dtype=torch.long)

        im_hid = self.model.dropout(self.model.relu(self.model.proj_h(features))).unsqueeze(0)
        im_state = self.model.dropout(self.model.relu(self.model.proj_c(features))).unsqueeze(0)
        h0 = torch.cat((im_hid, ) * self.model.num_layers, 0)                          # h0, c0: (num_layers, batch_size, hidden_dim)
        #h0 = torch.zeros((self.model.num_layers, N, self.model.hidden_dim)).to(device=device)
        c0 = torch.cat((im_state, ) * self.model.num_layers, 0)
        #c0 = torch.zeros(h0.size()).to(device=device)
        hidden = (h0, c0)

        features = self.model.dropout(self.model.relu(self.model.proj_f(features)))

        for t in range(max_length):
            word_scores, hidden = self.model.rnn(word_seq=feeds, im_feat=features,     # word_scores: (batch_size, 1, vocab_size)
                                                 hidden=hidden)
            feeds = torch.argmax(word_scores.squeeze(1), dim=1, keepdim=True)          # feeds: (batch_size, 1)

            ends_idx = np.where(feeds.cpu().data.numpy() == self._end)[0]
            if len(ends_idx) > 0:
                is_end[ends_idx] = True
            words_out_idx = feeds.cpu().data.numpy()
            words_out_idx[is_end] = self._pad
            captions[:, t] = words_out_idx.ravel()

        return captions

    def check_bleu(self, split, num_samples, batch_size=512, check_loss=False):
        captions, features, _ = sample_batch(self.data, num_samples, split)
        gt_captions = decode_captions(captions.numpy(), self.data['idx_to_word'])
        if check_loss:
            features = features.to(device=device)
            captions = captions.to(device=device)
            mask = (captions[:, 1:] != self._pad).view(-1)

            self.model.eval()  # set model to "evaluation" mode
            cap_input = captions[:, :-1]
            cap_target = captions[:, 1:].contiguous()
            scores, hidden = self.model(features, cap_input)
            loss = torch.sum(F.cross_entropy(scores.view(-1, scores.size(2))[mask],
                                             cap_target.view(-1)[mask],
                                             reduce=False)) / num_samples

        num_batches = num_samples // batch_size
        if num_samples % batch_size != 0:
            num_batches += 1
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

