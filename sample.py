from data_utils import *
from solver import *
from model import *

f_path = '/features/train_feat_arrays.pkl'
c_path = '/captions/train_cap_tokens.pkl'
images_path = '/flikr8k/'
model_path = '/model/'
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


def generate_captions(data, solver, sample_size):
    for split in ['train', 'val']:
        captions, features, im_ids = sample_batch(data, sample_size, split)
        gt_captions = decode_captions(captions.numpy(), data['idx_to_word'])

        sample_captions = solver.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, im_id in zip(gt_captions, sample_captions, im_ids):
            im = Image.open(images_path+'Flicker8k_Dataset/'+im_id)
            plt.imshow(np.array(im))
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig(output_path+'cap_%s.png' %(im_id,))


def main(args):
    data = load_features_tokens(f_path, c_path)

    model = CaptionModel(2048, 256, 256, 256, len(data['idx_to_word']), num_layers=1)
    solver = NetSolver(data, model)

    model.load_state_dict(torch.load(args.model_path))
    solver.model = model
    generate_captions(data, solver, args.sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=4,
                        help='sample size to generate')
    parser.add_argument('--model_path', type=str,
                        default=model_path+'img_caption_best.pth.tar')
    args = parser.parse_args()
    main(args)

