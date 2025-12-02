# =============================================
#  NNCUE: Efficiently Updatable Complex Neural 
#         Networks for Computer Chess
#      Using NNC (Complex Neural Network)
# ---------------------------------------------
#  By: Matheus Campos               12/02/2025
# =============================================

# This benchmark is a simplification. Understand
# how the architecture works if you need a more
# complete one.

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nnc import NNCUE_Network

BATCH_SIZE = 4096
STEPS = 2000
CSV_FILE = "positions.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NNUE_Standard(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_w = nn.EmbeddingBag(768, 512, mode='sum')
        self.hidden = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
    def forward(self, ix, off):
        h = torch.relu(self.hidden(self.in_w(ix,off)))
        return self.out(h)

def parse_eval(v):
    v = str(v).strip(); limit = 2000.0
    if "#" in v: val = limit if "+" in v else -limit
    else:
        try: val = float(v)
        except: val = 0.0
    return max(min(val, limit), -limit)

def fen_to_feats_list(fen):
    bd = fen.split()[0]; out = []; r = 7; f = 0
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
    for c in bd:
        if c == "/": r -= 1; f = 0
        elif c.isdigit(): f += int(c)
        else:
            p = piece_map.get(c)
            if p is not None: out.append(p*64 + (r*8 + f))
            f += 1
    return out

class ChessCSV(Dataset):
    def __init__(self, file):
        print(f"Loading {file}...")
        try:
            df = pd.read_csv(file)
            df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
            self.fens = df["FEN"].tolist()
            self.evals = [parse_eval(v)/400.0 for v in df["Evaluation"]]
            print(f"Loaded {len(self.fens)} positions.")
        except FileNotFoundError:
            print("Error: 'positions.csv' not found. Please add a dataset.")
            self.fens = []; self.evals = []

    def __len__(self): return len(self.fens)
    def __getitem__(self, i): return fen_to_feats_list(self.fens[i]), float(self.evals[i])

def collate(batch):
    if not batch: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    offs = [0]; feats = []; y = []; o = 0
    for f,v in batch:
        feats.extend(f); o += len(f); offs.append(o); y.append(v)
    return torch.tensor(feats,dtype=torch.long), torch.tensor(offs[:-1],dtype=torch.long), torch.tensor(y,dtype=torch.float32).unsqueeze(1)

def measure_speed(model, dl, name):
    model.eval().to(DEVICE)
    t0 = time.time(); total = 0
    with torch.no_grad():
        for i, (ix, off, y) in enumerate(dl):
            if i >= 100: break
            ix, off = ix.to(DEVICE), off.to(DEVICE)
            _ = model(ix, off)
            total += off.shape[0]
    dt = time.time() - t0
    print(f"[{name}] Throughput: {total/dt:,.0f} pos/sec")

def train_network(model, dl, name):
    print(f"Training {name}...")
    model.train().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    
    iter_dl = iter(dl)
    for i in range(STEPS):
        try: ix, off, y = next(iter_dl)
        except StopIteration: iter_dl = iter(dl); ix, off, y = next(iter_dl)
        
        ix, off, y = ix.to(DEVICE), off.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(ix, off)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        
        if i % 500 == 0: print(f"Step {i}: {loss.item():.4f}")
    return losses

if __name__ == "__main__":
    ds = ChessCSV(CSV_FILE)
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        
        nnue = NNUE_Standard()
        nncue = NNCUE_Network()
        
        print("\n=== SPEED TEST (CPU Simulation) ===")
        measure_speed(nnue, dl, "NNUE Standard")
        measure_speed(nncue, dl, "NNCUE v4.0")
        
        print("\n=== INTELLIGENCE TEST (Convergence) ===")
        l1 = train_network(nnue, dl, "NNUE Standard")
        l2 = train_network(nncue, dl, "NNCUE v4.0")
        
        plt.plot(l1, label="NNUE", alpha=0.5)
        plt.plot(l2, label="NNCUE v4.0", linewidth=2)
        plt.legend(); plt.title("Training Convergence"); plt.savefig("benchmark.png")
        print("\nDone. Results saved to benchmark.png")