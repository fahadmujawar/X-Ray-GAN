import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 128
#DATASET = 'D:\CAPSTONE\KUVAL DATASET\DATASET\preprocess_TUBERCULOSIS'
#DATASET = 'D:\CAPSTONE\CODE\PROGAN\Machine-Learning-Collection-master\ML\Pytorch\GANs\ProGAN\preprocessed_normal'
DATASET= 'D:/CAPSTONE/CODE/PROGAN/Machine-Learning-Collection-master/ML/Pytorch/GANs/ProGAN/new_tb2'
CHECKPOINT_GEN = "D:\CAPSTONE\CODE\PROGAN\Machine-Learning-Collection-master\ML\Pytorch\GANs\ProGAN\ProGAN_weights\generator.pth"
CHECKPOINT_CRITIC = "D:\CAPSTONE\CODE\PROGAN\Machine-Learning-Collection-master\ML\Pytorch\GANs\ProGAN\ProGAN_weights\critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE =  "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256  
IN_CHANNELS = 256  
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
#PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES) # for 128X128
# Adjust the PROGRESSIVE_EPOCHS list to train for more epochs at each resolution
#PROGRESSIVE_EPOCHS = [30, 30, 30, 30, 30, 30, 30, 30, 30]  # Example: 256x256
PROGRESSIVE_EPOCHS = [10, 10, 10, 10, 10, 10, 10, 10, 10]

FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4