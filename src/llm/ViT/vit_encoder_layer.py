import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.llm.ViT.vit_perceptron_block import VitPerceptronBlock
from src.llm.ViT.vit_multiheaded_self_attention import VitMultiHeadSelfAttentionBlock
from dotenv import load_dotenv
import os

ATTENTION_DROPOUT = int(os.getenv('ATTENTION_DROPOUT'))
MLP_DROPOUT = float(os.getenv('MLP_DROPOUT'))
MLP_SIZE = int(os.getenv('MLP_SIZE'))
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS'))
NUM_HEADS = int(os.getenv('NUMBER_OF_HEADS'))

class VitEncoder(nn.Module):
  def __init__(self, embedding_dims = EMBEDDING_DIMS,
               mlp_dropout=MLP_DROPOUT,
               attn_dropout=ATTENTION_DROPOUT,
               mlp_size = MLP_SIZE,
               num_heads = NUM_HEADS,
               ):
    super().__init__()

    self.msa_block = VitMultiHeadSelfAttentionBlock(embedding_dims = embedding_dims,
                                                 num_heads = num_heads,
                                                 attn_dropout = attn_dropout)

    self.mlp_block = VitPerceptronBlock(embedding_dims = embedding_dims,
                                                    mlp_size = mlp_size,
                                                    mlp_dropout = mlp_dropout,
                                                    )

  def forward(self,x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x

    return x