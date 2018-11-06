from data_utils import *

dtype = torch.cuda.FloatTensor
path = '/flikr8k/'
output_path = '/output/'


def preprocess(img):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def batchify(files):
    xs = []
    for im_id in files:
        im = Image.open(path+'Flicker8k_Dataset/'+im_id)
        x = preprocess(im)
        xs.append(x)
    return torch.cat(xs)


def encode_image(files, batch_size, spatial):
    encoded_images = {}

    net = models.resnet101(pretrained=True)
    net.type(dtype)
    net.eval()
    net_modules = list(net._modules.keys())[:-2] \
                    if spatial else list(net._modules.keys())[:-1]

    num_samples = len(files)
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0: num_batches += 1

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        xs = batchify(files[start:end]).type(dtype)

        for module_name in net_modules:
            xs = net._modules[module_name](xs)
        xs = xs.cpu().detach().numpy()
        if not spatial: xs = xs.squeeze()

        for j in range(xs.shape[0]):
            encoded_images[files[start:end][j]] = xs[j]

    return encoded_images


def main(args):
    trn_files, dev_files, test_files = split_image_files(path)
    train_features = encode_image(trn_files, args.batch_size, args.spatial)
    val_features = encode_image(dev_files, args.batch_size, args.spatial)
    test_features = encode_image(test_files, args.batch_size, args.spatial)
    print('number of train samples:', len(train_features))
    print('train features shape:', list(train_features.values())[0].shape)
    print('number of valid samples:', len(val_features))
    print('valid features shape:', list(val_features.values())[0].shape)
    print('number of test samples:', len(test_features))
    print('test features shape:', list(test_features.values())[0].shape)

    trn_feats, test_feats = {}, {}
    trn_feats['train_features'] = train_features
    trn_feats['val_features'] = val_features
    test_feats['test_features'] = test_features

    with open(output_path+args.train_feat_path, 'wb') as f:
        pickle.dump(trn_feats, f)
    with open(output_path+args.test_feat_path, 'wb') as f:
        pickle.dump(test_feats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_feat_path', type=str,
                        default='train_feat_arrays.pkl',
                        help='path for train image features file')
    parser.add_argument('--test_feat_path', type=str,
                        default='test_feat_array.pkl',
                        help='path for test image features file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size to transforming images')
    parser.add_argument('--spatial', type=bool, default=0,
                        help='whether to remain spatial info in features')
    args = parser.parse_args()
    main(args)

