import torch
from models.DNet import DNet
from models.LNet import LNet
from models.ENet import ENet


def _load(checkpoint_path):
    map_location=None if torch.cuda.is_available() else torch.device('cpu')
    return torch.load(checkpoint_path, map_location=map_location)

def load_checkpoint(path, model):
    print(f"Load checkpoint from: {path}")
    checkpoint = _load(path)
    s = checkpoint["state_dict"] if 'arcface' not in path else checkpoint
    new_s = {
        k.replace('module.', ''): v for k, v in s.items() if 'low_res' not in k
    }
    model.load_state_dict(new_s, strict=False)
    return model

def load_network(args):
    L_net = LNet()
    L_net = load_checkpoint(args.LNet_path, L_net)
    E_net = ENet(lnet=L_net)
    model = load_checkpoint(args.ENet_path, E_net)
    return model.eval()

def load_DNet(args):
    D_Net = DNet()
    print(f"Load checkpoint from: {args.DNet_path}")
    checkpoint =  torch.load(args.DNet_path, map_location=lambda storage, loc: storage)
    D_Net.load_state_dict(checkpoint['net_G_ema'], strict=False)
    return D_Net.eval()