import torch
from mamba_ssm import Mamba3

batch, length, dim = 2, 2048, 768
x = torch.randn(batch, length, dim).to(torch.bfloat16).to("cuda")

model = Mamba3(
    d_model=dim,
    d_state=128,
    headdim=64,
    is_mimo=True,
    mimo_rank=4,
    chunk_size=16,
    is_outproj_norm=False,
    dtype=torch.bfloat16,
).to("cuda")

y = model(x)
print(y.shape)