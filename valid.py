import time
from config import *
from util import AverageMeter, accuracy_top_k


def validate(val_loader, encoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param criterion: loss layer
    :return: classification accuaracy
    """
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    current_acc = AverageMeter()
    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, labels) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)

            # Calculate loss
            loss = criterion(imgs, labels)

            # Keep track of metrics
            classify_acc = accuracy_top_k(imgs, labels, top_k)
            losses.update(loss.item())
            current_acc.update(classify_acc)
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=current_acc))
        print('\n * LOSS - {loss.avg:.3f}, ACCURACY - {top5.avg:.3f}\n'.format(loss=losses,top5=current_acc))

    return current_acc.avg