import inspect
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
import pathlib
import os
from discrete_diffusion_scheduler import DiffusionTransformer
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from models.model2 import D2GraphTransformer
from datasets.zinc250k_dataset import Zin250KDataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default='data/zinc250k/')
parser.add_argument("--add-position", type=bool, default=True)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--num-layers", type=int, default=6)
parser.add_argument("--num-heads", type=int, default=8)
parser.add_argument("--drop-rate", type=float, default=0.0)
parser.add_argument("--last-drop-rate", type=float, default=0.0)

parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--diffusion_steps", type=int, default=300)
args = parser.parse_args()

def main():
    print(args)

    train_dataset = Zin250KDataset(stage='train',
                root=args.data_dir)
    test_dataset = Zin250KDataset(stage='test',
                root=args.data_dir)
    valid_dataset = Zin250KDataset(stage='val',
                root=args.data_dir)

    train_loader = DataLoader(train_dataset,
                batch_size=args.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset,
                batch_size=args.batch_size,collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                collate_fn=train_dataset.collate_fn)

    model = D2GraphTransformer(atom_dim=len(train_dataset.types),charge_dim=len(train_dataset.charges),edge_dim=len(train_dataset.bonds),
                 num_layers=args.num_layers,d_model=args.hidden_size,num_heads=args.num_heads,
                               dff=args.hidden_size,position=args.add_position)
    model = model.to(device=device)

    noise_scheduler = DiffusionTransformer(mask_id=0 ,a_classes=len(train_dataset.types),c_classes=len(train_dataset.charges),
                e_classes=len(train_dataset.bonds),model=model,diffusion_step=args.diffusion_steps,max_length=train_dataset.max_length)
    noise_scheduler = noise_scheduler.to(device)

    optimizer = torch.optim.AdamW(
        noise_scheduler.model.parameters(), lr=args.learning_rate, amsgrad=True,
        weight_decay=0
    )

    global_step = 0
    global_loss = 0

    for epoch in range(1,args.epochs+1):
        model.train()
        for step, batch in enumerate(train_loader):
            A, C, E  = batch
            A = A.to(device)
            C = C.to(device)
            E = E.to(device)

            diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(
                E.shape[0], -1, -1)
            E[diag] = 0

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            A_log_model_prob,C_log_model_prob, E_log_model_prob,loss = noise_scheduler(A,C,E)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            optimizer.zero_grad()

            global_loss = 0.99 * global_loss + 0.01 * loss.detach().cpu().item()
            global_step += 1

            if global_step % 500 == 0:
                print('epoch:', epoch, 'step:', global_step, 'train loss:', global_loss)
        if (epoch)%10==0:
            p = '_position' if args.add_position else ''
            if not os.path.exists(f'zinc250k_weights{p}/') :
                os.makedirs(f'zinc250k_weights{p}/')

            torch.save(noise_scheduler.model.state_dict(),f'zinc250k_weights{p}/model_weights_{epoch}.pt')

if __name__ == "__main__":
    main()
