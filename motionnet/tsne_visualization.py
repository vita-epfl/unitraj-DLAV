import os
import numpy as np
import torch
import wandb
from sklearn import manifold
from utils.config import load_config
from models import build_model
from motionnet.models.base_model.model_utils import draw_scene
from datasets import build_dataset
import pytorch_lightning as pl
from utils.tsne import visualize_tsne_points,visualize_tsne_images
import argparse

from torch.utils.data import DataLoader
import seaborn as sns

color = sns.color_palette("colorblind")

dataset_to_color = {
    'waymo': color[2],
    'nuplan': color[0],
    'av2': color[3],
    'nuscenes': color[4],
}

if __name__ == "__main__":
    tsne_tmp = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='cluster')
    parser.add_argument('--exp_name', '-e', default="test", type=str)
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--ckpt_path', '-p', type=str, default=None)
    parser.add_argument('--ratio', '-r', type=float, default=0.1)
    parser.add_argument('--image_size', '-i', type=int, default=500)
    parser.add_argument('--early_exaggeration', '-ee', type=float, default=12)
    parser.add_argument('--perpelxity', '-pp', type=float, default=30)
    args = parser.parse_args()
    cfg = load_config(args.config)

    model = build_model(cfg)
    train_set = build_dataset(cfg)

    # total_batch_size = cfg['data_loader']['batch_size']
    # batch_size = total_batch_size // len(args.devices)

    train_loader = DataLoader(
        train_set, batch_size=512, num_workers=cfg['data_loader']['load_num_workers'], shuffle=False,
        drop_last=False
    )

    if args.debug:
        trainer = pl.Trainer(
            devices=1, accelerator='cpu', profiler="simple", inference_mode=True
        )
    else:
        trainer = pl.Trainer(
            devices=1, accelerator='gpu', profiler="simple", inference_mode=True)

    predict = trainer.predict(model=model, dataloaders=train_loader, ckpt_path=args.ckpt_path)

    # concatentate all the predictions
    all_results = {}
    for i in range(len(predict)):
        for k in predict[i].keys():
            if k not in all_results:
                all_results[k] = []
            all_results[k].append(predict[i][k])
    for k,v in all_results.items():
        if isinstance(v[0],list):
            all_results[k] = sum(v, [])
        elif isinstance(v[0],torch.Tensor):
            all_results[k] = torch.cat(v, dim=0)
        else:
            all_results[k] = np.concatenate(v, axis=0)

    embeds = all_results['scene_emb']

    datasize = embeds.shape[0]
    point_num = datasize
    vis_num = int(point_num*args.ratio)

    dataset_list = all_results['dataset_name']
    # Draw a 3D TSNE
    tsne = manifold.TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        n_jobs=-1,
        early_exaggeration=args.early_exaggeration,
        perplexity=args.perpelxity
    )

    c_list = [dataset_to_color[c] for c in dataset_list]
    Y = tsne.fit_transform(embeds)

    ax = visualize_tsne_points(Y,c_list)
    tsne_points = wandb.Image(ax)

    # random select vis_num idx from point_num
    rand_indx = list(range(point_num))
    np.random.shuffle(rand_indx)
    rand_indx = rand_indx[:vis_num]

    c_list = np.array(c_list)[rand_indx]
    Y = Y[rand_indx]
    image_lsit = []
    for idx in rand_indx:
        ax = draw_scene(all_results['ego_full'][idx],all_results['other_full'][idx],all_results['map'][idx])
        # save
        os.makedirs('./img', exist_ok=True)
        save_path = os.path.join('./img', f'{idx}.png')
        ax.figure.savefig(save_path)
        image_lsit.append(save_path)

    tsne_image = wandb.Image(visualize_tsne_images(Y[:, 0], Y[:, 1], image_lsit, c_list,max_image_size=args.image_size))

    wandb.init(project='motionnet_vis',name=args.exp_name)
    wandb.log({"tsne_points": tsne_points,'tsne_image':tsne_image})