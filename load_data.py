import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorflow.keras.utils import to_categorical

from optimizer import PlainRAdam

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def evaluate():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            if i + batch_size < len(test_data):
                print(i, batch_size)
                inputs = ((test_data[i: i + batch_size]).astype(dtype=np.float) / 255 - 0.5) * 2
                inputs = np.reshape(inputs, (batch_size, 3, 32, 32))
                labels = train_labels[i: i + batch_size]
            else:
                inputs = ((test_data[i: len(test_data)]).astype(dtype=np.float) / 255 - 0.5) * 2
                inputs = np.reshape(inputs, (batch_size, 3, 32, 32))
                labels = train_labels[i: len(test_data)]
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %f %%' % (float(correct) / total))


if __name__ == "__main__":
    """show it works"""

    net = VGG('VGG16')
    # PATH = './cifar_net.pth'
    # net.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = PlainRAdam(net.parameters())

    cifar_10_dir = 'data/cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)
    batch_size = 100

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        for i in range(0, len(train_data), batch_size):
            inputs = []
            labels = []
            if i + batch_size < len(train_data):
                inputs = ((train_data[i: i + batch_size]).astype(dtype=np.float) / 255 - 0.5) * 2
                inputs = np.reshape(inputs, (batch_size, 3, 32, 32))
                labels = train_labels[i: i + batch_size]
            else:
                inputs = ((train_data[i: len(train_data)]).astype(dtype=np.float) / 255 - 0.5) * 2
                inputs = np.reshape(inputs, (batch_size, 3, 32, 32))
                labels = train_labels[i: len(train_data)]
            # labels = to_categorical(labels)

            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels)

            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total += 1

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / total))

        evaluate()

        torch.save(net.state_dict(), PATH)

    print('Finished Training')


    # plt.imshow(train_data[1])
    # plt.show()

    # # Don't forget that the label_names and filesnames are in binary and need conversion if used.
    # 
    # # display some random training images in a 25x25 grid
    # num_plot = 5
    # f, ax = plt.subplots(num_plot, num_plot)
    # for m in range(num_plot):
    #     for n in range(num_plot):
    #         idx = np.random.randint(0, train_data.shape[0])
    #         ax[m, n].imshow(train_data[idx])
    #         ax[m, n].get_xaxis().set_visible(False)
    #         ax[m, n].get_yaxis().set_visible(False)
    # f.subplots_adjust(hspace=0.1)
    # f.subplots_adjust(wspace=0)
    # plt.show()