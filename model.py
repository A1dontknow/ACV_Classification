import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        self.resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, 91)

        self.fine_tune()

    # Output is probabilities vector of 91 classes in COCO dataset
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune