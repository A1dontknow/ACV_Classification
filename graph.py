import numpy as np
from matplotlib import pyplot as plt

# Hard code from training log
train_acc = [47.628, 54.702, 56.395, 57.402, 58.014, 58.688, 59.381, 59.610, 59.885, 60.355, 60.594, 60.795, 61.253, 61.508, 61.646, 62.013, 62.230, 62.389, 62.569, 62.975, 62.997, 63.343, 63.560, 63.784, 63.891, 64.000, 64.118, 64.783, 65.125, 65.247, 65.507, 65.610, 65.679, 65.875, 65.973, 66.011, 66.420, 66.794, 67.075, 67.184, 67.359, 67.295, 67.557, 67.604, 67.918, 68.417, 68.352, 68.578, 68.728]
valid_acc = [57.046, 59.315, 59.554, 59.435, 59.912, 60.470, 60.569, 60.490, 60.490, 60.748, 60.788, 61.266, 61.385, 60.947, 61.505, 61.226, 61.664, 61.365, 61.883, 61.624, 61.286, 61.266, 61.226, 61.724, 61.843, 61.385, 61.186, 61.783, 62.221, 61.963, 61.604, 61.764, 61.744, 60.729, 61.505, 61.923, 61.903, 61.166, 62.002, 61.764, 61.963, 62.002, 61.684, 61.664, 61.724, 62.321, 61.724, 61.923, 61.803]
train_loss = [1.9623, 1.6193, 1.5439, 1.5001, 1.4659, 1.4381, 1.4181, 1.3977, 1.3805, 1.3586, 1.3488, 1.3362, 1.3208, 1.3082, 1.2965, 1.2832, 1.2735, 1.2619, 1.2533, 1.2441, 1.2361, 1.2265, 1.2148, 1.2060, 1.1977, 1.1939, 1.1848, 1.1595, 1.1480, 1.1390, 1.1327, 1.1268, 1.1226, 1.1112, 1.1073, 1.0987, 1.0913, 1.0749, 1.0682, 1.0588, 1.0538, 1.0474, 1.0435, 1.0422, 1.0303, 1.0162, 1.0096, 1.0047, 0.9993]
valid_loss = [1.527, 1.456, 1.428, 1.416, 1.399, 1.397, 1.381, 1.365, 1.370, 1.367, 1.353, 1.348, 1.361, 1.370, 1.363, 1.370, 1.340, 1.349, 1.359, 1.360, 1.375, 1.365, 1.354, 1.359, 1.339, 1.368, 1.370, 1.347, 1.348, 1.362, 1.374, 1.351, 1.355, 1.370, 1.369, 1.359, 1.373, 1.385, 1.371, 1.365, 1.358, 1.372, 1.382, 1.383, 1.391, 1.378, 1.376, 1.380, 1.376]


if __name__ == '__main__':
    no_batch = len(train_acc)
    epoch_list = np.arange(no_batch) + 1
    _, ax = plt.subplots(1, 2)
    ax[0].plot(epoch_list, train_acc, marker='o', label='Train')
    ax[0].plot(epoch_list, valid_acc, marker='o', label='Valid')
    ax[0].set_title('Train and valid accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].grid(True)

    ax[1].plot(epoch_list, train_loss, marker='o', label='Train')
    ax[1].plot(epoch_list, valid_loss, marker='o', label='Valid')
    ax[1].set_title('Train and valid loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].grid(True)

    plt.suptitle('Train result')
    plt.show()