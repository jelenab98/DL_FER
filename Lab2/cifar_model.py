from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from pathlib import Path
from torch import nn
from utils import *
import matplotlib.pyplot as plt
import torch


DATA_DIR = Path(__file__).parent / 'datasets/cifar-10-batches-py'
SAVE_DIR = Path(__file__).parent / 'out_4'
BATCH_SIZE = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def create_cifar10_loaders():
    transform_operation = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(DATA_DIR, transform=transform_operation)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
    test_dataset = CIFAR10(DATA_DIR, train=False, transform=transform_operation)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    """
    Moguće je napraviti i tensor dataloadere preko donje naredbe i korištenjem
    
    train_loader = DataLoader(TensorDataset(train_x))
    
    """
    return train_loader, valid_loader, test_loader


def create_cifar10_numpy():

    img_height = 32
    img_width = 32
    num_channels = 3
    num_classes = 10

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, data_mean, data_std


class MyModel(nn.Module):
    """
    Promjena veličina tenzora po slojevima:
        mapa pri izlazu iz konvolucije: [(broj_ulaza - jezgra +2*padding) / stride] + 1
        mapa pri izlazu iz poolinga: [(velicina_ulaza +2*padding - dilation * (jezgra - 1) - 1 ) / stride + 1]

    Parametri:
        konvolucije: ((jezgra_h * jezgra_w * in_channels) + 1) * out_channels
        potpuno povezani: ulazni * izlazni + izlazni

    Receptivno polje:
        r_ulaz + (jezgra - 1) * skok
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), bias=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), bias=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear) and m is not self.fc3:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        self.fc3.reset_parameters()
        return

    def forward(self, x):
        # 1x16x32x32
        h = self.conv1(x)
        # 1x16x28x28
        h = torch.relu(h)
        h = self.max_pool1(h)
        # 1x16x13x13
        h = self.conv2(h)
        # 1x32x9x9
        h = torch.relu(h)
        h = self.max_pool2(h)
        # 1x32x4x4
        h = h.view(h.shape[0], -1)
        # 1x512
        h = self.fc1(h)
        h = torch.relu(h)
        # 1x256
        h = self.fc2(h)
        h = torch.relu(h)
        # 1x128
        logits = self.fc3(h)
        # 1x10
        return logits

    def get_loss(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        logits = self.forward(x)
        max_logits, _ = torch.max(logits, dim=1)
        max_logits = max_logits.unsqueeze(dim=1)
        max_logits = max_logits.repeat(1, logits.shape[1])
        logits = logits - max_logits
        log_sum = torch.log(torch.sum(torch.exp(logits), dim=-1) + 1e-15)
        mul_sum = torch.sum(logits * y, dim=-1)
        first_loss = torch.mean(log_sum - mul_sum)
        second_loss = -torch.mean(torch.sum(y * torch.log(torch.softmax(logits, dim=1) + 1e-15), dim=1))
        return first_loss

    def get_loss_by_examples(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        logits = self.forward(x)
        max_logits, _ = torch.max(logits, dim=1)
        max_logits = max_logits.unsqueeze(dim=1)
        max_logits = max_logits.repeat(1, logits.shape[1])
        logits = logits - max_logits
        log_sum = torch.log(torch.sum(torch.exp(logits), dim=-1) + 1e-15)
        mul_sum = torch.sum(logits * y, dim=-1)
        first_loss = log_sum - mul_sum
        return first_loss


def train(model, train_loader, valid_loader, optimizer, scheduler, criterion, epochs):
    for epoch in range(1, epochs + 1):
        losses_train = []
        accuracy_train = 0
        # training the model
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            predictions = logits.data.max(1, keepdim=True)[1]

            loss = criterion(logits, targets)
            loss.backward()

            optimizer.step()

            losses_train.append(loss.item())
            accuracy_train += predictions.eq(targets.data.view_as(predictions)).sum()
            if idx % 5 == 0:
                print(f"epoch {epoch}, step {idx * BATCH_SIZE}/{len(train_loader.dataset)}, "
                      f"batch loss = {loss.item():.2f}")

        print(f"Train accuracy = {accuracy_train / len(train_loader.dataset):.2f}")

        # validation
        model.eval()
        losses_valid = []
        accuracy_valid = 0
        print("Running evaluation: Validation")
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(valid_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                predictions = logits.data.max(1, keepdim=True)[1]

                loss = criterion(logits, targets)
                losses_valid.append(loss.item())
                accuracy_valid += predictions.eq(targets.data.view_as(predictions)).sum()

        # printing statistics
        accuracy_train = accuracy_train / len(train_loader.dataset)
        accuracy_valid = accuracy_valid / len(valid_loader.dataset)
        loss_train = sum(losses_train) / len(train_loader)
        loss_valid = sum(losses_valid) / len(valid_loader)

        plot_data['train_loss'] += [loss_train]
        plot_data['valid_loss'] += [loss_valid]
        plot_data['train_acc'] += [accuracy_train]
        plot_data['valid_acc'] += [accuracy_valid]
        plot_data['lr'] += [scheduler.get_lr()]

        print(f"Validation accuracy = {accuracy_valid:.2f}")
        print(f"Validation avg loss = {loss_valid:.2f}")

        with (SAVE_DIR / "train_stats.txt").open(mode="a") as f:
            f.write(f"{accuracy_train} {loss_train}\n")
        with (SAVE_DIR / "valid_stats.txt").open(mode="a") as f:
            f.write(f"{accuracy_valid} {loss_valid}\n")
        with (SAVE_DIR / "lr.txt").open(mode="a") as f:
            f.write(f"{scheduler.get_lr()}\n")

        show_filters(epoch, model.conv1, SAVE_DIR)
        show_images(model, valid_loader, epoch, SAVE_DIR)
        scheduler.step()
    plot_training_progress(SAVE_DIR, plot_data)


def get_worst_images(model, x, y, y_classes, std=0.5, mi=0.5):
    model.eval()
    with torch.no_grad():
        y_predicted = np.argmax(model.forward(torch.FloatTensor(x)).detach().cpu().numpy(), axis=1)
        loss = model.get_loss(x, y)

    worst_20_values, worst_20_indices = torch.topk(loss, 20)
    fig = plt.figure(figsize=(15, 10))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        idx = worst_20_indices[i]
        loss_value = worst_20_values[i]
        y_pred = y_predicted[idx]
        y_gt = y_classes[idx]
        img = x[idx]
        img *= std
        img += mi
        plt.imshow(img)
        plt.title(f"Y:{y_pred}, GT:{y_gt}")

    plt.show()


def get_class_stats(model, x, y, y_classes):
    loss, acc, p, r, f1, class_stats = evaluate(model, x, y, y_classes)
    f1_classes = class_stats["f1"]
    sorted_f1_classes = np.argsort(f1_classes)
    highest_f1 = sorted_f1_classes[-3:]
    for h in highest_f1:
        print(f"{h} {f1_classes[h]}")


def evaluate(model, x, y, y_classes):
    model.eval()
    with torch.no_grad():
        y_predicted = np.argmax(model.forward(torch.FloatTensor(x)).detach().cpu().numpy(), axis=1)
        loss = model.get_loss(x, y)

    conf_matrix = confusion_matrix(y_classes, y_predicted)
    num_classes = conf_matrix.shape[0]
    num_correct = conf_matrix.trace()
    total_examples = conf_matrix.sum()

    accuracy = num_correct / total_examples

    tps = np.diag(conf_matrix)
    tp_fn = np.sum(conf_matrix, axis=1)
    tp_fp = np.sum(conf_matrix, axis=0)

    classes_stats = {"p": np.zeros(num_classes), "r": np.zeros(num_classes), "f1": np.zeros(num_classes)}
    for i in range(len(num_classes)):
        precision = tps[i, i] / tp_fp[i]
        recall = tps[i, i] / tp_fn[i]
        f1 = (2 * precision * recall) / (precision + recall)
        classes_stats["p"][i] = precision
        classes_stats["r"][i] = recall
        classes_stats["f1"][i] = f1

    precision = classes_stats["p"].mean()
    recall = classes_stats["r"].mean()
    f1 = classes_stats["f1"].mean()

    return loss, accuracy, precision, recall, f1, classes_stats


def run(model, num_epochs, train_x, train_labels, train_y, valid_x, valid_labels,
        valid_y, n_batch, optimizer, lr_scheduler):
    for epoch in range(num_epochs):
        X, Yoh = shuffle_data(train_x, train_labels)
        X = torch.FloatTensor(X)
        Yoh = torch.FloatTensor(Yoh)
        for batch in range(n_batch):
            # broj primjera djeljiv s veličinom grupe bsz
            batch_X = X[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :]
            batch_Yoh = Yoh[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE, :]

            loss = model.get_loss(batch_X, batch_Yoh)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch%100 == 0:
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, n_batch, loss))

        train_loss, train_acc, train_p, train_r, train_f, _ = evaluate(model, train_x, train_labels, train_y)
        val_loss, val_acc, val_p, val_r, val_f, _ = evaluate(model, valid_x, valid_labels, valid_y)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [lr_scheduler.get_lr()]

        lr_scheduler.step()
        show_filters(epoch, model.conv1, SAVE_DIR)

    plot_training_progress(SAVE_DIR, plot_data)


if __name__ == '__main__':
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []
    train_x, train_y, valid_x, valid_y, test_x, test_y = create_cifar10_numpy()
    train_y_oh = dense_to_one_hot(train_y, 10)
    valid_y_oh = dense_to_one_hot(valid_y, 10)
    test_y_oh = dense_to_one_hot(test_y, 10)
    model = MyModel()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    run(model, 40, train_x, train_y_oh, train_y, valid_x, valid_y_oh, valid_y,
        len(train_x) // BATCH_SIZE, optimizer, scheduler)
    """
    # data loaders
    train_loader, valid_loader, test_loader = create_cifar10_loaders()

    # model
    model = MyModel().to(device)
    print(model.parameters)
    show_filters(0, model.conv1, SAVE_DIR)
    show_images(model, test_loader, 0, SAVE_DIR)
    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)
    
    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, valid_loader, optimizer, scheduler, criterion, 8)
    """