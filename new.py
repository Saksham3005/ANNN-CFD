# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# import torch.optim as optim
# from tqdm import tqdm
# from torch.optim.lr_scheduler import StepLR
# import os
# from torchsummary import summary
# import sys
# # Constants


# nu = 0.01

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# weights_folder = "W_and_B/final_params"

# path_joined = []
# for j in range(0, 25):
#     path_joined.append(f"use/data_new{j}.csv")
    


# class DATA(Dataset):
#     def __init__(self, paths, transform=None):
#         self.paths = paths
#         self.transform = transform
#         df = pd.read_csv(paths)

#     #     # Compute min and max for normalization
#     #     self.u_min, self.u_max = df['u'].min(), df['u'].max()
#     #     self.v_min, self.v_max = df['v'].min(), df['v'].max()
#     #     self.p_min, self.p_max = df['p'].min(), df['p'].max()
        
#         self.df = df

#     # def normalize(self, values, min_val, max_val):
#     #     return (values - min_val) / (max_val - min_val)

#     def __len__(self):
#         return int(len(self.df)/1000) - 1

#     def __getitem__(self, idx):
#         ui = self.df.iloc[idx*1000 : (1+idx)*1000]['u']
#         vi = self.df.iloc[idx*1000 : (1+idx)*1000]['v']
#         pi = self.df.iloc[idx*1000 : (1+idx)*1000]['p']
#         bi = self.df.iloc[idx*1000 : (1+idx)*1000]['boundary']
        
#         x = self.df.iloc[idx*1000 : (1+idx)*1000]['x']
#         y = self.df.iloc[idx*1000 : (1+idx)*1000]['y']

#         i = idx+1

#         uf = self.df.iloc[i*1000 : (1+i)*1000]['u']
#         vf = self.df.iloc[i*1000 : (1+i)*1000]['v']
#         pf = self.df.iloc[i*1000 : (1+i)*1000]['p']
#         bf = self.df.iloc[i*1000 : (1+i)*1000]['boundary']

#         # Apply min-max normalization
#         # ui = self.normalize(ui, self.u_min, self.u_max)
#         # vi = self.normalize(vi, self.v_min, self.v_max)
#         # pi = self.normalize(pi, self.p_min, self.p_max)

#         # uf = self.normalize(uf, self.u_min, self.u_max)
#         # vf = self.normalize(vf, self.v_min, self.v_max)
#         # pf = self.normalize(pf, self.p_min, self.p_max)

#         sample = {
#             # 'ui': torch.tensor(ui.to_numpy(), dtype=torch.float32),
#             # 'vi': torch.tensor(vi.to_numpy(), dtype=torch.float32),
#             # 'pi': torch.tensor(pi.to_numpy(), dtype=torch.float32),
#             'bi': torch.concat([torch.tensor([-999.99],dtype=torch.float32),torch.tensor(bi.to_numpy(), dtype=torch.float32)], dim=0),
#             'uf': torch.tensor(uf.to_numpy(), dtype=torch.float32),
#             'vf': torch.tensor(vf.to_numpy(), dtype=torch.float32),
#             'pf': torch.tensor(pf.to_numpy(), dtype=torch.float32),
#             # 'bf': torch.tensor(bf.to_numpy(), dtype=torch.float32),
#             'x' : torch.tensor(x.to_numpy(), dtype=torch.float32),
#             'y' : torch.tensor(y.to_numpy(), dtype=torch.float32)
#         }

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


    
# for path in path_joined:
    
#     PINN_dataset = DATA(path)
#     data = DataLoader(PINN_dataset, batch_size=1, pin_memory=True, num_workers)
    