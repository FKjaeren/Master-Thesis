import os
import torch
import torchrec
import torch.distributed as dist

eb_configs = [
   EmbeddingBagConfig(
       name=f"t_{feature_name}",
       embedding_dim=64,
       num_embeddings=100_000,
       feature_names=[feature_name],
   )
   for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
]

from torchrec.modules.fused_embedding_modules import fuse_embedding_optimizer
ebc = fuse_embedding_optimizer(
    ebc,
    optimizer_type=torch.optim.SGD,
    optimizer_kwargs={"lr": 0.02},
    device=torch.device("meta"),
)