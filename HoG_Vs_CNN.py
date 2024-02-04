import numpy as np
import torch
import torch.nn as nn
import cv2
from keras_preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def cosine_similarity(v1: np.array,v2: np.array):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def Hog_features(img):
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9

    winSize = (img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0])

    blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])

    blockStride = (cell_size[1], cell_size[0])

    hog = cv2.HOGDescriptor(_winSize=winSize,
                            _blockSize=blockSize,
                            _blockStride=blockStride,
                            _cellSize=cell_size,
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])

    hog_feats = hog.compute(img) \
        .reshape(n_cells[1] - block_size[1] + 1,
                 n_cells[0] - block_size[0] + 1,
                 block_size[0], block_size[1], nbins) \
        .transpose((1, 0, 2, 3, 4))
    return hog_feats.flatten()


def CNN_features(model_no_fc, img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img / 255.)
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    a = model_no_fc(img).squeeze().cpu().numpy()
    return model_no_fc(img).view(-1).cpu().numpy()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('coco/Resnet101_fc_dr50_fc_wei/BEST_checkpoint_coco.pth.tar')['encoder']
    # very cursed command...
    model_no_fc = nn.Sequential(*list(list(model.children())[0].children())[:-2])
    model_no_fc = model_no_fc.to(device)
    model_no_fc.eval()

    img = load_img('horse1.jpg', target_size=(256, 256))
    img = img_to_array(img).astype(np.uint8)
    # hog = Hog_features(img.astype(np.uint8))

    img2 = load_img('laugh2.jpg', target_size=(256, 256))
    img2 = img_to_array(img2).astype(np.uint8)
    # hog2 = Hog_features(img2.astype(np.uint8))

    cnn_img1 = CNN_features(model_no_fc, img)
    cnn_img2 = CNN_features(model_no_fc, img2)
    hog_img1 = Hog_features(img)
    hog_img2 = Hog_features(img2)


    _, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].axis('off')
    plt.suptitle('Resnet similarity = ' + str(cosine_similarity(cnn_img1, cnn_img2)) \
                 + "\n\nHoG similarity = " + str(cosine_similarity(hog_img1, hog_img2)))
    plt.show()