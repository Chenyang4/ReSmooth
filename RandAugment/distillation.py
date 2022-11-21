import torch

from RandAugment.networks import get_model, num_class


def get_teacher(args, path=''):
    if not path:
        path = args.save.split('/')
        path[-2] = 'baseline'
        path = '/'.join(path)

    print("Distillation Initialization\nLoading Teacher Model:", path)
    net = get_model(args.model, num_class(args.dataset))
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.cuda()
    print('Done')
    return net