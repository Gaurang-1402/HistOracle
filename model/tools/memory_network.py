import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import pdb

class Memory_Network(nn.Module):
    
    def __init__(self, mem_size, color_feat_dim = 512, spatial_feat_dim = 512, top_k = 256, alpha = 0.1, age_noise = 4.0, gpu_ids = []):
        
        super(Memory_Network, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.ResNet18 = ResNet18().to(self.device)
        self.ResNet18 = self.ResNet18.eval()
        self.mem_size = mem_size
        self.color_feat_dim = color_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.alpha = alpha
        self.age_noise = age_noise
        self.top_k = top_k
        
        ## Each color_value is probability distribution
        self.color_value = F.normalize(random_uniform((self.mem_size, self.color_feat_dim), 0, 0.01), p = 1, dim=1).to(self.device)
        
        self.spatial_key = F.normalize(random_uniform((self.mem_size, self.spatial_feat_dim), -0.01, 0.01), dim=1).to(self.device)
        self.age = torch.zeros(self.mem_size).to(self.device)
        
        self.top_index = torch.zeros(self.mem_size).to(self.device)
        self.top_index = self.top_index - 1.0
        
        self.color_value.requires_grad = False
        self.spatial_key.requires_grad = False