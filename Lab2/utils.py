import os
import pickle
import math
import numpy as np
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import torch


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def show_filters(epoch, layer, save_dir):
    weight = layer.weight.detach().cpu().numpy()
    weights = weight.copy()
    weights -= weights.min()
    weights /= weights.max()
    for i in range(16):
        plt.subplot(2, 8, i+1)
        image = weights[i, :, :]
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(i)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.savefig(save_dir/f"kernels_conv1_epoch{epoch}.png")


def show_images(model, test_loader, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(test_loader))
        for i in range(16):
            plt.subplot(2, 8, i+1)
            image = inputs[i, :, :, :]
            image = torch.unsqueeze(image, dim=0)
            prediction = model(image).data.max(1, keepdims=True)[1]
            image = image.squeeze()
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            image *= 0.5
            image += 0.5
            plt.imshow(image)
            plt.title(f"y_ = {targets[i]}")
            plt.xlabel(f"y = {prediction.detach().cpu().numpy()[0, 0]}")
            plt.xticks([])
            plt.yticks([])
    plt.savefig(save_dir / f"images_epoch{epoch}.png")
    plt.show()


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    plt.imshow(img)
    plt.show()
    ski.io.imsave(os.path.join(save_dir, filename), img)


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)
