import cv2 as cv
import torch
from torch import nn
import os
from dotenv import load_dotenv
from src.llm.ViT.vit_transformer import VitTransformer

load_dotenv()

IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS'))
NUM_OF_PATCHES = int(os.getenv('NUM_OF_PATCHES'))
PATCH_SIZE = int(os.getenv('PATCH_SIZE')) 

def get_image_embedding(training_image):


    # convert the image to a tensor
    training_image =  torch.tensor(training_image, dtype=torch.float32).permute(2, 0, 1)

    # create a convolutional layer
    conv_layer = nn.Conv2d(in_channels = IMAGE_CHANNELS, out_channels = EMBEDDING_DIMS, kernel_size = PATCH_SIZE, stride = PATCH_SIZE)

    # Pass the image through the convolution layer
    image_through_conv = conv_layer(training_image.unsqueeze(0))

    # Permute the dimensions of image_through_conv to match the expected shape
    image_through_conv = image_through_conv.permute((0, 2, 3, 1))

    # Create a flatten layer using nn.Flatten
    flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

    # Pass the image_through_conv through the flatten layer
    image_through_conv_and_flatten = flatten_layer(image_through_conv)

    # Assign the embedded image to a variable
    embedded_image = image_through_conv_and_flatten

    # calss token embedding
    class_token_embeddings = nn.Parameter(torch.rand((1, 1,EMBEDDING_DIMS), requires_grad  = True))

    # patch embeddings with class token
    embedded_image_with_class_token_embeddings = torch.cat((class_token_embeddings, embedded_image), dim = 1)

    # positional embeddings for patches
    position_embeddings = nn.Parameter(torch.rand((1, NUM_OF_PATCHES+1, EMBEDDING_DIMS ), requires_grad = True ))

    # get the final embeddings
    final_embeddings = embedded_image_with_class_token_embeddings + position_embeddings

    # create image embeddings
    image_embeddings = VitTransformer(num_classes = 3)

    # pass image throught the transformer
    image_embeddings = image_embeddings(training_image.unsqueeze(0))

    # convert to list
    image_embedding = image_embeddings[0].tolist()


    return image_embedding