# Data parameters
import torch
from torch.backends import cudnn

data_folder = 'D:\ACV_True_Final\classify\\'  # folder with data files saved by create_input_files.py
data_name = 'coco'  # base name shared by data files

# Model parameters
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = 'coco/shufflenet_v2_1.0_fc/checkpoint_coco.pth.tar'  # 'coco/BEST_checkpoint_coco.pth.tar'  # path to checkpoint, None if none
best_acc = 0.  # Best classification

# Evaluate parameter
top_k = 1  # Accuracy top-k
