import torch
def unit_difference(full,idk):
 v=(full-idk).mean(0);return v/(v.norm()+1e-12)
def random_like(v,n=20,seed=42):
 g=torch.Generator().manual_seed(seed);x=torch.randn((n,v.numel()),generator=g);return x/x.norm(dim=1,keepdim=True)
