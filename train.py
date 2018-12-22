from data_utils import *
from solver import *
from model import *

trn_feat_path = '/features/train_feat_arrays.pkl'
trn_cap_path = '/captions/train_cap_tokens.pkl'
test_feat_path = '/features/test_feat_array.pkl'
test_cap_path = '/captions/test_cap_tokens.pkl'

image_path = '/flikr8k/'
model_path = '/model/'
output_path = '/output/'


def load_features_tokens(feat_path, cap_path, max_train=False):
    with open(cap_path, 'rb') as f:
        data = pickle.load(f)
    with open(feat_path, 'rb') as f:
        feats = pickle.load(f)
    for k in feats.keys():
        data[k] = feats[k]
    if max_train:
        size = data['train_captions'].shape[0]
        mask = np.random.choice(size, max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_ids'] = data['train_image_ids'][mask]
        trn_encoded = {id:data['train_features'][id] for id in data['train_image_ids']}
        data['train_features'] = trn_encoded
    return data

def load_augmented_features_tokens(trn_feat_path, trn_cap_path, test_feat_path, test_cap_path):
    data = {}

    with open(trn_cap_path, 'rb') as f:
        trn_caps = pickle.load(f)
    with open(test_cap_path, 'rb') as f:
        test_caps = pickle.load(f)
    trn_caps['train_captions'] = np.concatenate([trn_caps['train_captions'], trn_caps['val_captions']])
    trn_caps['train_image_ids'] = np.concatenate([trn_caps['train_image_ids'], trn_caps['val_image_ids']])
    trn_caps['val_captions'] = test_caps['test_captions']
    trn_caps['val_image_ids'] = test_caps['test_image_ids']
    for k in trn_caps.keys():
        data[k] = trn_caps[k]

    with open(trn_feat_path, 'rb') as f:
        trn_feats = pickle.load(f)
    with open(test_feat_path, 'rb') as f:
        test_feats = pickle.load(f)
    data['train_features'] = trn_feats['train_features']
    for im_id in trn_feats['val_features']:
        data['train_features'][im_id] = trn_feats['val_features'][im_id]
    data['val_features'] = test_feats['test_features']

    return data


def main(args):
    if args.test_model:
        # test output tensor shape
        caption_model_test((2048,), 40)
        return

    if args.overfit:
        cp_name = 'overfit_model.pth.tar'
        data = load_features_tokens(f_path, c_path, max_train=50)
        model = CaptionModel_B(2048, 128, 160, len(data['idx_to_word']), num_layers=1)
        solver = NetSolver(data, model, lr_init=5e-3, lr_decay=0.995, print_every=2,
                           batch_size=25, checkpoint_name=cp_name)
        t0 = time.time()
        solver.train(epochs=50)
        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
        return

    if args.pre_encoded:
        data = load_features_tokens(trn_feat_path, trn_cap_path)
        #data = load_augmented_features_tokens(trn_feat_path, trn_cap_path, test_feat_path, test_cap_path)
        vocab_size = len(data['idx_to_word'])
    else:
        loader_train = prepare_loader(image_path, batch_size=args.batch_size, data_part='train')
        loader_val = prepare_loader(image_path, batch_size=128, data_part='val')
        data = loader_val.dataset
        vocab_size = len(data.idx_to_word)

    if args.test_solver:
        # test check_bleu
        model = CaptionModel_B(2048, 128, 160, vocab_size, num_layers=1)
        solver = NetSolver(data, model)
        solver.check_bleu('train', num_samples=2000, check_loss=1)

    if args.train:
        #model = CaptionModel(2048, 128, 100, 200, vocab_size, num_layers=1)
        #model = AttentionModel(2048, 128, 160, vocab_size)
        model = CaptionModel_B(2048, 128, 160, vocab_size, num_layers=1)
        if args.preload:
            model.load_state_dict(torch.load(model_path+args.model_name))
        solver = NetSolver(data, model, batch_size=args.batch_size)

        loaders = None if args.pre_encoded else (loader_train, loader_val)
        t0 = time.time()
        solver.train(args.epochs, loaders=loaders, pre_encoded=args.pre_encoded)
        #train_bleu = solver.check_bleu(loader_train, num_batches=62)
        #print('Train BLEU:', train_bleu)
        #val_loss, val_bleu = solver.check_bleu(loader_val, num_batches=15, check_loss=True)
        #print('Val LOSS:', val_loss, 'Val BLEU:', val_bleu)
        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        if args.log_plot:
            history = [solver.bleu_history, solver.val_bleu_history, solver.loss_history, solver.val_loss_history]
            plot_history(history, output_path+'im_cap_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-model', type=bool, default=0,
                        help='test model output')
    parser.add_argument('--test-solver', type=bool, default=0,
                        help='test solver\'s evaluation output')
    parser.add_argument('--overfit', type=bool, default=0,
                        help='check if the model learns')
    parser.add_argument('--train', type=bool, default=0,
                        help='whether to train a model')
    parser.add_argument('--log-plot', type=bool, default=1,
                        help='whether to plot the metrics log')
    parser.add_argument('--pre-encoded', type=bool, default=0,
                        help='whether to pre-encode image data')
    parser.add_argument('--preload', type=bool, default=0,
                        help='whether to pre-load model')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    args = parser.parse_args()
    main(args)

