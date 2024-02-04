from config import *
from torch import nn
from torchvision.transforms import transforms
import torch.utils.data
from dataset import ClassifyDataset
from valid import validate
from util import get_number_parameter

if __name__ == '__main__':
    test_path = 'coco/vgg16_to_fc/BEST_checkpoint_coco.pth.tar'

    checkpoint = torch.load(test_path)
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    print("Number of parameters =", get_number_parameter(encoder))
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        ClassifyDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    validate(val_loader=test_loader,
             encoder=encoder,
             criterion=criterion)
