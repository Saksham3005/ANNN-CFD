import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
from torchsummary import summary
import sys
from model import Net, LSTMNet
from dataset import DATA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights_folder = "W_and_B_LSTM_2"
os.makedirs(weights_folder,exist_ok=True)
n = 0
folders = sorted(os.listdir(weights_folder))
for folder in folders:
    m =  int(folder.split("_")[2])
    if(m>n):
        n=m
n+=1    
weights_path = f"{weights_folder}/PINN_attempt_{n-1}_best.pt"
    
path_joined = []
for j in range(0, 25):
    path_joined.append(f"use/data_new{j}.csv")

model = LSTMNet().to(device)
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path), strict=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
max_epochs = 300
batch = 999
best_loss = 99999
  
PINN_dataset = DATA("use/data_new24.csv")
data = DataLoader(PINN_dataset, batch_size=batch, pin_memory=True, num_workers=2, drop_last=True)  

for epoch in tqdm(range(max_epochs)):
    print(f"Epoch : {epoch+1}")
    optimizer.zero_grad()
    total_loss = 0
        
    for i, sample in enumerate(data):
        sample = sample.to(device)
        input = torch.hstack((sample[:,7000:8000],sample[:,8000:9000],sample[:,0:1000], sample[:,1000:2000], sample[:,3000:4000])).to(device)
        # input = torch.squeeze(input, dim=0).to(device)
        output = model(input)
        
        if output.shape[0] != 3000:
            u = output[:,torch.arange(0, 3000, 3)]
            v = output[:,torch.arange(1, 3000, 3)]
            p = output[:,torch.arange(2, 3000, 3)]
        else:
            u = output[torch.arange(0, 3000, 3)]
            v = output[torch.arange(1, 3000, 3)]
            p = output[torch.arange(2, 3000, 3)]

        loss_u_next = criterion(u, sample[:,4000:5000])
        loss_v_next = criterion(v, sample[:,5000:6000])
        loss_p_next = criterion(p, sample[:,6000:7000])

        loss = loss_u_next + loss_v_next + loss_p_next
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
    
    if(total_loss < best_loss):
        best_loss = total_loss
        PATH = f'{weights_folder}/PINN_attempt_{n}_best.pt'
        torch.save(model.state_dict(), PATH)

    if (epoch+1) % 40 == 0:
        PATH = f'{weights_folder}/PINN_attempt_{n}_{epoch+1}.pt'
        torch.save(model.state_dict(), PATH)
        print(f"Saving Weights : f'{weights_folder}/PINN_attempt_{n}_{epoch+1}.pt'")            