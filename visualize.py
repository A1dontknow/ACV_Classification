import torch
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

if __name__ == '__main__':

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    img_path = 'D:\ACV_True_Final'
    checkpoint = torch.load('coco/Resnet101_fc_dr50_fc_wei/BEST_checkpoint_coco.pth.tar')
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    encoder = checkpoint['encoder']
    best_acc = checkpoint['best_acc']
    encoder_optimizer = checkpoint['encoder_optimizer']

    img = load_img(img_path + '/coco/train2014/COCO_train2014_000000392136.jpg', target_size=(256, 256))
    img = img_to_array(img).transpose(2, 0, 1)
    img = torch.FloatTensor(img / 255.)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    normalized_img = normalize(img)
    img = normalized_img.cuda()
    encoder.eval()
    img = torch.unsqueeze(img, dim=0)
    classify = encoder(img)
    max_index = torch.argmax(classify, dim=1)
    plt.imshow(normalized_img.cpu().numpy().transpose(1, 2, 0))
    plt.title(CLASSES[max_index[0].item()])
    plt.axis('off')
    plt.show()
