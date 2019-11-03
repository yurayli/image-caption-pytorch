from .data_utils import *
from .solver import *
from .model import *

trn_feat_path = '/features/train_feat_arrays.pkl'
trn_cap_path = '/captions/train_cap_tokens.pkl'
test_feat_path = '/features/test_feat_array.pkl'
test_cap_path = '/captions/test_cap_tokens.pkl'

image_path = '/flikr8k/'
model_path = '/model/'
output_path = '/output/'


def main(args):
    # output tensor shape test
    if args.test_model:
        caption_model_test((2048,), 40)
        return

    # overfitting test
    if args.overfit:
        cp_name = 'overfit_model.pth.tar'
        data = load_features_tokens(trn_feat_path, trn_cap_path,
                                    test_feat_path, test_cap_path, max_train=50)
        model = CaptionModel_B(2048, 50, 160, len(data['idx_to_word']), num_layers=1)
        solver = NetSolver(data, model, lr_init=5e-3, lr_decay=0.995, print_every=2,
                           batch_size=25, checkpoint_name=cp_name)
        t0 = time.time()
        solver.train(epochs=50)
        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
        return

    # load data
    if args.pre_encoded:
        data = load_features_tokens(trn_feat_path, trn_cap_path, test_feat_path, test_cap_path)
        vocab_size = len(data['idx_to_word'])
    else:
        loader_train = prepare_loader(image_path, batch_size=args.batch_size, data_part='train')
        loader_val = prepare_loader(image_path, batch_size=128, data_part='test')
        data = loader_val.dataset
        vocab_size = len(data.idx_to_word)

    # check_bleu test
    if args.test_solver:
        model = CaptionModel_B(2048, 50, 160, vocab_size, num_layers=1)
        solver = NetSolver(data, model)
        solver.check_bleu('train', num_samples=2000, check_loss=1)

    # model training
    if args.train:
        model = CaptionModel_B(2048, 50, 160, vocab_size, num_layers=1)
        if args.preload:
            model.load_state_dict(torch.load(model_path+args.model_name))
        solver = NetSolver(data, model, lr_init=args.lr, batch_size=args.batch_size)

        loaders = None if args.pre_encoded else (loader_train, loader_val)
        t0 = time.time()
        solver.train(args.epochs, loaders=loaders, pre_encoded=args.pre_encoded)
        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        if args.log_plot:
            history = [solver.bleu_history, solver.val_bleu_history, solver.loss_history, solver.val_loss_history]
            with open('history.pkl', 'wb') as f:
                pickle.dump(history, f)
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
    parser.add_argument('--pre-encoded', type=bool, default=1,
                        help='whether to pre-encode image data')
    parser.add_argument('--preload', type=bool, default=0,
                        help='whether to pre-load model')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size during training')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    args = parser.parse_args()
    main(args)

