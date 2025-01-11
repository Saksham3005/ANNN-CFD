from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import sys
import os
import matplotlib.pyplot as plt

class DATA(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.df = pd.read_csv(file_path)
        
    def __len__(self):
        return int(len(self.df)/1000) - 1 

    def __getitem__(self, idx):
        ui = self.df.iloc[idx*1000 : (1+idx)*1000]['u']
        vi = self.df.iloc[idx*1000 : (1+idx)*1000]['v']
        pi = self.df.iloc[idx*1000 : (1+idx)*1000]['p']
        bi = self.df.iloc[idx*1000 : (1+idx)*1000]['boundary']
        
        x = self.df.iloc[idx*1000 : (1+idx)*1000]['x']
        y = self.df.iloc[idx*1000 : (1+idx)*1000]['y']

        i = idx+1

        uf = self.df.iloc[i*1000 : (1+i)*1000]['u']
        vf = self.df.iloc[i*1000 : (1+i)*1000]['v']
        pf = self.df.iloc[i*1000 : (1+i)*1000]['p']

        sample = [
            torch.tensor(ui.to_numpy(), dtype=torch.float32),
            torch.tensor(vi.to_numpy(), dtype=torch.float32),
            torch.tensor(pi.to_numpy(), dtype=torch.float32),
            torch.tensor(bi.to_numpy(), dtype=torch.float32),
            torch.tensor(uf.to_numpy(), dtype=torch.float32),
            torch.tensor(vf.to_numpy(), dtype=torch.float32),
            torch.tensor(pf.to_numpy(), dtype=torch.float32),
            torch.tensor(x.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy(), dtype=torch.float32)
        ]
        
        sample = torch.concat(sample,dim=0)

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__=="__main__":
    pinn_dataset = DATA("use/data_new24.csv")
    batch = 999
    loader = DataLoader(pinn_dataset,batch_size=batch, drop_last=True)
    # t = torch.ones((batch, 1000), dtype=torch.float32)
    for i, sample in enumerate(loader):
        print(sample.shape)
        # input = torch.hstack((sample[:,7000:8000], sample[:,8000:9000], t, sample[:,3000:4000]))
    #     b = sample[:,3000:4000]
    #     b_flatten = torch.flatten(b)
    #     b_new = torch.reshape(b_flatten,(999,1000))
    # plt.imshow(b_new,cmap='hot')
    # plt.show()