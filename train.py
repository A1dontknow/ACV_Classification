import time
from config import *
from util import AverageMeter, clip_gradient, accuracy_top_k

def train(train_loader, encoder, criterion, encoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param epoch: epoch number
    """
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    current_acc = AverageMeter()
    start = time.time()

    # Batches
    for i, (imgs, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # Calculate loss
        loss = criterion(imgs, labels)

        # Additional regularization? No for now
        # loss += 0   #alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        classify_acc = accuracy_top_k(imgs, labels, top_k)
        losses.update(loss.item())
        current_acc.update(classify_acc)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses,
                                                                    top5=current_acc))