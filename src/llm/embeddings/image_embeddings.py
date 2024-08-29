import cv2 as cv
import torch
from torch import nn
import os
from dotenv import load_dotenv
from src.llm.transformers.image.transformer import VitTransformer

load_dotenv()

IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS'))
NUM_OF_PATCHES = int(os.getenv('NUM_OF_PATCHES'))
PATCH_SIZE = int(os.getenv('PATCH_SIZE')) 

def get_image_embedding(training_image):

    # convert the image to a tensor
    training_image =  torch.tensor(training_image, dtype=torch.float32).permute(2, 0, 1)

    # create image embeddings
    image_embeddings = VitTransformer(num_classes = 3)

    # pass image throught the transformer
    image_embeddings = image_embeddings(training_image.unsqueeze(0))

    # convert to list
    image_embedding = image_embeddings[0].tolist()

    return image_embedding
