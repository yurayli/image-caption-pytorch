from data_utils import *
from solver import *
from model import *
from train import *

trn_feat_path = '/features/train_feat_arrays.pkl'
trn_cap_path = '/captions/train_cap_tokens.pkl'
test_feat_path = '/features/test_feat_array.pkl'
test_cap_path = '/captions/test_cap_tokens.pkl'

image_path = '/flikr8k/'
model_path = '/model/'
output_path = '/output/'


def generate_captions(data, solver, sample_size):
    for split in ['val']:
        captions, features, im_ids = sample_batch(data, sample_size, split)
        gt_captions = decode_captions(captions.numpy(), data['idx_to_word'])

        sample_caps_1 = solver.sample(features, search_mode='greedy')
        sample_caps_1 = decode_captions(sample_caps_1, data['idx_to_word'])
        sample_caps_2 = solver.sample(features, search_mode='beam', b_size=3)
        sample_caps_2 = decode_captions(sample_caps_2, data['idx_to_word'])
        sample_caps_3 = solver.sample(features, search_mode='beam', b_size=5)
        sample_caps_3 = decode_captions(sample_caps_3, data['idx_to_word'])
        sample_caps_4 = solver.sample(features, search_mode='beam', b_size=7)
        sample_caps_4 = decode_captions(sample_caps_4, data['idx_to_word'])

        sample_caps = (gt_captions, sample_caps_1, sample_caps_2, sample_caps_3,
                       sample_caps_4, im_ids)

        for gt_cap, s1, s2, s3, s4, im_id in zip(*sample_caps):
            im = Image.open(image_path+'Flicker8k_Dataset/'+im_id)
            plt.figure(figsize=(10,10))
            plt.imshow(np.array(im))
            plt.title('Greedy:%s\nB=3:%s\nB=5:%s\nB=7:%s\nGT:%s' % (s1, s2, s3, s4, gt_cap))
            plt.axis('off')
            plt.savefig(output_path+'cap_%s.png' %(im_id,))


def main(args):
    data = load_features_tokens(trn_feat_path, trn_cap_path)
    #data = load_augmented_features_tokens(trn_feat_path, trn_cap_path, test_feat_path, test_cap_path)

    model = CaptionModel_B(2048, 128, 160, len(data['idx_to_word']), num_layers=1)
    solver = NetSolver(data, model)

    model.load_state_dict(torch.load(model_path+args.model_name))
    solver.model = model

    generate_captions(data, solver, args.sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=4,
                        help='sample size to generate')
    parser.add_argument('--model-name', type=str)
    args = parser.parse_args()
    main(args)

