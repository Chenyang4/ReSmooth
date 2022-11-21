import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from RandAugment.networks import get_model, num_class


class GMM:
    def __init__(self, net, loader):
        self.net = net
        self.net.eval()

        loss_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                data_ori, data_aug, label = batch[0] + [batch[1], ]
                data_ori, data_aug, label = data_ori.cuda(), data_aug.cuda(), label.cuda()

                preds = self.net(data_ori)
                loss = F.cross_entropy(preds, label, reduction='none')
                loss_list.append(loss)

                preds = self.net(data_aug)
                loss = F.cross_entropy(preds, label, reduction='none')
                loss_list.append(loss)

            loss_list = torch.log(torch.cat(loss_list, dim=0) + 1e-10)

        self.gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        self.gmm.fit(loss_list.view(-1, 1).cpu().numpy())

    def predict(self, inputs, labels):
        with torch.no_grad():
            preds = self.net(inputs)
            input_loss = F.cross_entropy(preds, labels, reduction='none')
        prob = self.gmm.predict_proba(torch.log(input_loss + 1e-10).view(-1, 1).cpu().numpy())
        prob = prob[:, self.gmm.means_.argmin()]
        return prob


def get_gmm(args, loader, path=''):
    if not path:
        path = args.save.split('/')
        path[-2] = 'baseline'
        path = '/'.join(path)

    print("GMM Initialization\nLoading Model:", path)
    net = get_model(args.model, num_class(args.dataset))
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.cuda()

    gmm = GMM(net, loader)
    print('Done')
    return gmm