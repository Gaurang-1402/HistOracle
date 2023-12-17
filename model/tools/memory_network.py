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

        self.Linear = nn.Linear(512, spatial_feat_dim)
        self.body = [self.ResNet18, self.Linear]
        self.body = nn.Sequential(*self.body)
        self.body = self.body.to(self.device)

        def forward(self, x):
            q = self.body(x)
            q = F.normalize(q, dim=1)
            return q

        def unsupervised_loss(self, query, color_feat, color_thres):
        
            bs = query.size()[0]
            cosine_score = torch.matmul(query, torch.t(self.spatial_key))
            
            top_k_score, top_k_index = torch.topk(cosine_score, self.top_k, 1)
            
            ### For unsupervised training
            color_value_expand = torch.unsqueeze(torch.t(self.color_value), 0)
            color_value_expand = torch.cat([color_value_expand[:,:,idx] for idx in top_k_index], dim = 0)
            
            color_feat_expand = torch.unsqueeze(color_feat, 2)
            color_feat_expand = torch.cat([color_feat_expand for _ in range(self.top_k)], dim = 2)
            
            #color_similarity = self.KL_divergence(color_value_expand, color_feat_expand, 1)
            color_similarity = torch.sum(torch.mul(color_value_expand, color_feat_expand),dim=1)
            
            #loss_mask = color_similarity < color_thres
            loss_mask = color_similarity > color_thres
            loss_mask = loss_mask.float()
            
            pos_score, pos_index = torch.topk(torch.mul(top_k_score, loss_mask), 1, dim = 1)
            neg_score, neg_index = torch.topk(torch.mul(top_k_score, 1 - loss_mask), 1, dim = 1)
            
            loss = self._unsupervised_loss(pos_score, neg_score)
            
            return loss

        