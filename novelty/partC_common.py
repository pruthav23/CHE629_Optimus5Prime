import numpy as np
import torch
from data import build_chem_table, train_val
from models import CNN
from train import predict, Scaler

t = build_chem_table()
U_train = np.where(t.split == "train")[0]
YU = t.rl_U.values


def pretrained(seed):
    m = CNN(dropout=0.2)
    m.load_state_dict(torch.load(f"results/novelty/partA/single_U_s{seed}.pt"))
    f, _ = train_val(U_train, seed, 0.10)
    return m, Scaler(YU[f])


def mrl(model, sc, x):
    return np.clip(sc.inv(predict(model, x)), 0.1, None)
