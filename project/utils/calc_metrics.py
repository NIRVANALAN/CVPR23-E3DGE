from argparse import ArgumentParser
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from project.core.losses.builder import LossClass

# sys.path.append(".")
# sys.path.append("..")

from project.data.gt_res_dataset import GTResDataset


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode',
                        type=str,
                        default='lpips',
                        choices=['lpips', 'l2'])
    parser.add_argument('--data_path', type=str, default='results')
    parser.add_argument('--gt_path', type=str, default='gt_images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args


def run(args):

    device = 'cuda'

    opt = dict(l2_lambda=1, vgg_lambda=1, id_lambda=1)
    loss_func = LossClass(device, opt).to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print('Loading dataset')
    dataset = GTResDataset(root_path=args.data_path,
                           gt_dir=args.gt_path,
                           transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=int(args.workers),
                            drop_last=True)

    def _calc_average_loss(all_loss_dicts):
        all_scores = {}  # todo, defaultdict
        mean_all_scores = {}

        for loss_dict in all_loss_dicts:
            for k, v in loss_dict.items():
                v = v.item()
                if k not in all_scores:
                    # all_scores[f'{k}_val'] = [v]
                    all_scores[k] = [v]
                else:
                    all_scores[k].append(v)

        for k, v in all_scores.items():
            mean = np.mean(v)
            std = np.std(v)
            if k in ['loss_lpis', 'loss_ssim']:
                mean = 1 - mean
            result_str = '{} average loss is {:.4f} +- {:.4f}'.format(
                k, mean, std)
            mean_all_scores[k] = mean
            print(result_str)

        val_scores_for_logging = {
            f'{k}_val': v
            for k, v in mean_all_scores.items()
        }
        return val_scores_for_logging

    global_i = 0
    scores_dict = {}
    all_scores = []
    all_loss_dicts = []

    for result_batch, gt_batch in tqdm(dataloader):
        # for i in range(args.batch_size):

        _, loss_2d_rec_dict = loss_func.calc_2d_rec_loss(result_batch,
                                                         gt_batch,
                                                         gt_batch,
                                                         opt,
                                                         loss_dict=True,
                                                         mode='val')  # 21.9G

        #=================== log metric =================
        loss_dict = {**loss_2d_rec_dict}
        all_loss_dicts.append(loss_dict)

        # im_path = dataset.pairs[global_i][0]
        # scores_dict[os.path.basename(im_path)] = loss
        global_i += 1

    val_scores_for_logging = _calc_average_loss(all_loss_dicts)

    print('Finished with ', args.data_path)
    # print(result_str)

    out_path = os.path.join(os.path.dirname(args.data_path),
                            'inference_metrics')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # with open(os.path.join(out_path, 'stat_{}.txt'.format(args.mode)), 'w') as f:
    # 	f.write(result_str)
    # with open(os.path.join(out_path, 'scores_{}.json'.format(args.mode)), 'w') as f:
    # 	json.dump(scores_dict, f)


if __name__ == '__main__':
    args = parse_args()
    run(args)
