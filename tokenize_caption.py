from data_utils import *

path = '/flikr8k/'
output_path = '/output/'


def word_idx_map(raw_captions, threshold):
    caps = []
    for im in raw_captions:
        for s in raw_captions[im]:
            caps.append(s.split())

    word_freq = nltk.FreqDist(itertools.chain(*caps))
    idx_to_word = ['<pad>'] + [word for word, cnt in word_freq.items() if cnt >= threshold] + ['<unk>']
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}

    return idx_to_word, word_to_idx


def tokenize(captions, word_to_idx, maxlen):
    '''
    Inputs:
    - captions: dictionary with image_id as key, captions as value
    - word_to_idx: mapping from word to index
    - maxlen: max length of each sequence of tokens

    Returns:
    - tokens: array of shape (data_size, maxlen)
    - image_ids: list of length data_size, mapping token to corresponding image_id
    '''
    tokens, image_ids = [], []
    for im_id in captions:
        for cap in captions[im_id]:
            token = [(lambda x: word_to_idx[x] if x in word_to_idx else word_to_idx['<unk>'])(w) \
                     for w in cap.split()]
            if len(token) > maxlen:
                token = token[:maxlen]
            else:
                token += [0] * (maxlen-len(token))
            tokens.append(token)
            image_ids.append(im_id)
    return np.array(tokens).astype('int32'), np.array(image_ids)



def main(args):
    trn_files, dev_files, test_files = split_image_files(path)
    trn_raw_captions, dev_raw_captions, test_raw_captions, raw_captions = \
        split_captions(path, trn_files, dev_files, test_files)
    idx_to_word, word_to_idx = word_idx_map(trn_raw_captions, args.threshold)

    trn_captions, trn_image_ids = tokenize(trn_raw_captions, word_to_idx, args.maxlen)
    dev_captions, dev_image_ids = tokenize(dev_raw_captions, word_to_idx, args.maxlen)
    test_captions, test_image_ids = tokenize(test_raw_captions, word_to_idx, args.maxlen)

    trn_data, test_data = {}, {}
    trn_data['train_captions'] = trn_captions
    trn_data['train_image_ids'] = trn_image_ids
    trn_data['val_captions'] = dev_captions
    trn_data['val_image_ids'] = dev_image_ids
    trn_data['idx_to_word'] = idx_to_word
    trn_data['word_to_idx'] = word_to_idx
    test_data['test_captions'] = test_captions
    test_data['test_image_ids'] = test_image_ids

    with open(output_path+args.train_cap_path, 'wb') as f:
        pickle.dump(trn_data, f)
    with open(output_path+args.test_cap_path, 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cap_path', type=str,
                        default='train_cap_tokens.pkl',
                        help='path for train annotation file')
    parser.add_argument('--test_cap_path', type=str,
                        default='test_cap_tokens.pkl',
                        help='path for test annotation file')
    parser.add_argument('--threshold', type=int, default=1,
                        help='minimum word count threshold')
    parser.add_argument('--maxlen', type=int, default=40,
                        help='maximum sequence length')
    args = parser.parse_args()
    main(args)


