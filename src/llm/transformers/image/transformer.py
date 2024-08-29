import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os
from  src.llm.transformers.image.patch_embeddings import PatchEmbeddingLayer
from  src.llm.transformers.image.image_encoder import VitEncoder

load_dotenv()

ATTENTION_DROPOUT = int(os.getenv('ATTENTION_DROPOUT'))
MLP_DROPOUT = float(os.getenv('MLP_DROPOUT'))
MLP_SIZE = int(os.getenv('MLP_SIZE'))
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS'))
NUM_HEADS = int(os.getenv('NUMBER_OF_HEADS'))
IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH'))
IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
PATCH_SIZE = int(os.getenv('PATCH_SIZE'))
NUMBER_OF_TRANSFORMER_LAYERS = int(os.getenv('NUMBER_OF_LAYERS'))
DEFAULT_NO_OF_CLASSES = int(os.getenv('DEFAULT_NO_OF_CLASSES'))

class VitTransformer(nn.Module):
  def __init__(self, img_size = IMAGE_WIDTH,
               in_channels = IMAGE_CHANNELS ,
               patch_size = PATCH_SIZE,
               embedding_dims = EMBEDDING_DIMS,
               num_transformer_layers = NUMBER_OF_TRANSFORMER_LAYERS, # from table 1 above
               mlp_dropout = MLP_DROPOUT,
               attn_dropout = ATTENTION_DROPOUT,
               mlp_size = MLP_SIZE,
               num_heads = NUM_HEADS,
               num_classes = DEFAULT_NO_OF_CLASSES):
    
    super().__init__()

    self.patch_embedding_layer = PatchEmbeddingLayer(in_channels = in_channels,
                                                     patch_size=patch_size,
                                                     embedding_dim = embedding_dims)

    self.transformer_encoder = nn.Sequential(*[VitEncoder(embedding_dims = embedding_dims,
                                              mlp_dropout = mlp_dropout,
                                              attn_dropout = attn_dropout,
                                              mlp_size = mlp_size,
                                              num_heads = num_heads) for _ in range(num_transformer_layers)])

  def forward(self, x):
    return self.transformer_encoder(self.patch_embedding_layer(x))[:, 0]