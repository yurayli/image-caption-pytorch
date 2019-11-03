from .data_utils import *

dtype = torch.cuda.FloatTensor
path = '/flikr8k/'
output_path = '/output/'


def preprocess(img, augment=False):
    if augment:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
    else:
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


def feed_forward_net(xs, net, pre_net, spatial=False):

    if pre_net == 'inception_v3':
        for module_name in list(net._modules.keys())[:3]:
            xs = net._modules[module_name](xs)
        xs = F.max_pool2d(xs, kernel_size=3, stride=2)
        for module_name in list(net._modules.keys())[3:5]:
            xs = net._modules[module_name](xs)
        xs = F.max_pool2d(xs, kernel_size=3, stride=2)
        for module_name in list(net._modules.keys())[5:-5]:
            xs = net._modules[module_name](xs)
        for module_name in list(net._modules.keys())[-4:-1]:
            xs = net._modules[module_name](xs)
        xs = F.avg_pool2d(xs, kernel_size=8)
        return xs.squeeze()

    if pre_net == 'densenet161':
        xs = net.features(xs)
        xs = F.relu(xs, inplace=True)
        xs = F.avg_pool2d(xs, kernel_size=7, stride=1).view(xs.size(0), -1)
        return xs.squeeze()

    if pre_net == 'resnet101':
        l = 2 if spatial else 1
        net_modules = list(net._modules.keys())[:-l]
        for module_name in net_modules:
            xs = net._modules[module_name](xs)
        return xs if spatial else xs.squeeze()

    if pre_net == 'vgg16':
        xs = net.features(xs)
        if spatial:
            return xs
        else:
            xs = xs.view(xs.size(0), -1)
            xs = net.classifier(xs)
            return xs


def encode_image(files, batch_size, spatial, pre_net):
    if pre_net == 'inception_v3':
        net = models.inception_v3(pretrained=True)
    if pre_net == 'densenet161':
        net = models.densenet161(pretrained=True)
    if pre_net == 'resnet101':
        net = models.resnet101(pretrained=True)
    if pre_net == 'vgg16':
        net = models.vgg16_bn(pretrained=True)
    net.type(dtype)
    net.eval()

    encoded_images = {}

    num_samples = len(files)
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0: num_batches += 1

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        xs = batchify(files[start:end]).type(dtype)

        xs = feed_forward_net(xs, net, pre_net, spatial)
        xs = xs.cpu().detach().numpy()

        for j in range(xs.shape[0]):
            encoded_images[files[start:end][j]] = xs[j]

    return encoded_images



def main(args):
    trn_files, test_files = split_image_files(path)
    train_features = encode_image(trn_files, args.batch_size, args.spatial, args.pre_net)
    test_features = encode_image(test_files, args.batch_size, args.spatial, args.pre_net)
    print('number of train samples:', len(train_features))
    print('train features shape:', list(train_features.values())[0].shape)
    print('number of test samples:', len(test_features))
    print('test features shape:', list(test_features.values())[0].shape)

    trn_feats, test_feats = {}, {}
    trn_feats['train_features'] = train_features
    test_feats['test_features'] = test_features

    with open(output_path+args.train_feat_path+'.pkl', 'wb') as f:
        pickle.dump(trn_feats, f)
    with open(output_path+args.test_feat_path+'.pkl', 'wb') as f:
        pickle.dump(test_feats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_feat_path', type=str,
                        default='train_feat_arrays',
                        help='path for train image features file')
    parser.add_argument('--test_feat_path', type=str,
                        default='test_feat_array',
                        help='path for test image features file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size to transforming images')
    parser.add_argument('--spatial', type=bool, default=0,
                        help='whether to remain spatial info in features')
    parser.add_argument('--pre-net', type=str, default='resnet101',
                        help='which pretrained cnn to use')
    args = parser.parse_args()
    main(args)

