import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()

ATTENTION_DROPOUT = int(os.getenv('ATTENTION_DROPOUT'))
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS'))
NUM_HEADS = int(os.getenv('NUM_HEADS'))

class VitMultiHeadSelfAttentionBlock(nn.Module):
  def __init__(self,
               embedding_dims = EMBEDDING_DIMS, # Hidden Size D in the ViT Paper Table 1
               num_heads = NUM_HEADS,  # Heads in the ViT Paper Table 1
               attn_dropout = ATTENTION_DROPOUT # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
               ):
    super().__init__()

    self.embedding_dims = embedding_dims
    self.num_head = num_heads
    self.attn_dropout = attn_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)

    self.multiheadattention =  nn.MultiheadAttention(num_heads = NUM_HEADS,
                                                     embed_dim = EMBEDDING_DIMS,
                                                     dropout = attn_dropout,
                                                     batch_first = True,
                                                    )

  def forward(self, x):
    x = self.layernorm(x)
    output,_ = self.multiheadattention(query=x, key=x, value=x,need_weights=False)
    return output