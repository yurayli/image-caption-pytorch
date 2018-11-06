from data_utils import *
from solver import *
from model import *

f_path = '/features/train_feat_arrays.pkl'
c_path = '/captions/train_cap_tokens.pkl'
output_path = '/output/'


def load_features_tokens(feat_path, cap_path, max_train=None):
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


def main(args):
    if args.test_model:
        # test output tensor shape
        caption_model_test((2048,), 40)
        return

    if args.overfit:
        cp_name = 'overfit_model.pth.tar'
        data = load_features_tokens(f_path, c_path, max_train=50)
        model = CaptionModel(2048, 512, 256, 512, len(data['idx_to_word']))
        solver = NetSolver(data, model, lr_init=5e-3, lr_decay=0.995, print_every=2,
                           batch_size=25, checkpoint_name=cp_name)
        t0 = time.time()
        solver.train(epochs=50)
        print('Training goes with %.2f seconds.' %(time.time()-t0))
        return

    data = load_features_tokens(f_path, c_path)

    if args.test_solver:
        # test check_bleu
        model = CaptionModel(2048, 512, 50, 100, len(data['idx_to_word']))
        solver = NetSolver(data, model)
        solver.check_bleu('train', num_samples=2000, check_loss=1)

    if args.train:
        model = CaptionModel(2048, 256, 256, 256, len(data['idx_to_word']), num_layers=1)
        solver = NetSolver(data, model, batch_size=args.batch_size, checkpoint_name=args.cp_name)

        t0 = time.time()
        solver.train(epochs=args.epochs)
        print('Training goes with %.2f seconds.' %(time.time()-t0))

        if args.log_plot:
            history = [solver.bleu_history, solver.val_bleu_history, solver.loss_history, solver.val_loss_history]
            plot_history(history, output_path+'im_cap_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', type=bool, default=0,
                        help='test model output')
    parser.add_argument('--test_solver', type=bool, default=0,
                        help='test solver\'s evaluation output')
    parser.add_argument('--overfit', type=bool, default=0,
                        help='check if the model learns')
    parser.add_argument('--train', type=bool, default=0,
                        help='whether to train a model')
    parser.add_argument('--log_plot', type=bool, default=1,
                        help='whether to plot the metrics log')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--cp_name', type=str,
                        default='img_caption_best.pth.tar')
    args = parser.parse_args()
    main(args)

