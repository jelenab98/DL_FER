from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt
import torch
import pandas as pd


DATA_DIR = Path(__file__).parent / 'datasets'
SAVE_DIR = Path(__file__).parent / 'out_3'
BATCH_SIZE = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mnist_loaders():
    transform_operation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
    train_dataset = MNIST(DATA_DIR, transform=transform_operation)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
    test_dataset = MNIST(DATA_DIR, train=False, transform=transform_operation)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, valid_loader, test_loader


def show_filters(epoch, layer):
    weights = layer.weight.squeeze()
    weights = weights.detach().cpu().numpy()
    weights -= weights.min()
    weights /= weights.max()
    for i in range(16):
        plt.subplot(2, 8, i+1)
        image = weights[i, :, :]
        plt.imshow(image)
        plt.title(i)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(SAVE_DIR/f"kernels_conv1_epoch{epoch}.png")
    plt.show()


def show_images(model, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(test_loader))
        preds = model(inputs).data.max(1, keepdims=True)[1]
        preds = preds.data.view_as(targets)
        print(preds)
        for i in range(16):
            plt.subplot(2, 8, i+1)
            image = inputs[i, :, :, :]
            image = torch.unsqueeze(image, dim=0)
            prediction = model(image).data.max(1, keepdims=True)[1]
            image = image.squeeze()
            image = image.detach().cpu().numpy()
            image *= 0.3081
            image += 0.1307
            plt.imshow(image)
            plt.title(f"y_ = {targets[i]}")
            plt.xlabel(f"y = {prediction.detach().cpu().numpy()[0, 0]}")
            plt.xticks([])
            plt.yticks([])
    plt.savefig(SAVE_DIR / f"images_epoch{epoch}.png")
    plt.show()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=(2, 2), bias=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2), bias=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear) and m is not self.fc2:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        self.fc2.reset_parameters()
        return

    def forward(self, x):
        # 1x1x28x28
        h = self.conv1(x)
        # 1x16x28x28
        h = self.max_pool1(h)
        # 1x16x14x14
        h = torch.relu(h)

        h = self.conv2(h)
        # 1x32x14x14
        h = self.max_pool2(h)
        # 1x32x7x7
        h = torch.relu(h)

        h = h.view(h.shape[0], -1)
        # 1x1568
        h = self.fc1(h)
        h = torch.relu(h)
        # 1x512
        logits = self.fc2(h)
        # 1x10
        return logits


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

        print(f"Validation accuracy = {accuracy_valid:.2f}")
        print(f"Validation avg loss = {loss_valid:.2f}")

        with (SAVE_DIR / "train_stats.txt").open(mode="a") as f:
            f.write(f"{accuracy_train} {loss_train}\n")
        with (SAVE_DIR / "valid_stats.txt").open(mode="a") as f:
            f.write(f"{accuracy_valid} {loss_valid}\n")

        show_filters(epoch, model.conv1)
        show_images(model, valid_loader, epoch)
        scheduler.step()


def test(model, test_loader, criterion):
    model.eval()
    loss = 0
    accuracy = 0
    print("Running evaluation: Test")
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            predictions = logits.data.max(1, keepdim=True)[1]

            loss_tmp = criterion(logits, targets)

            loss += loss_tmp.item()
            accuracy += predictions.eq(targets.data.view_as(predictions)).sum()

    loss /= len(test_loader)
    acc = accuracy / len(test_loader.dataset)

    print(f"Test accuracy = {acc:.2f}")
    print(f"Test avg loss = {loss:.2f}")
    show_images(model, test_loader, "final")


def show_loss(train_loss, valid_loss):
    train_stats = pd.read_csv(train_loss, header=None, sep=" ")
    valid_stats = pd.read_csv(valid_loss, header=None, sep=" ")
    figure = plt.figure(figsize=(15, 10))
    t_loss = train_stats[1]
    v_loss = valid_stats[1]
    plt.plot(range(len(t_loss)), t_loss)
    plt.plot(range(len(v_loss)), v_loss)
    plt.show()


if __name__ == '__main__':
    # data loaders
    train_loader, valid_loader, test_loader = create_mnist_loaders()

    # model
    model = MyModel().to(device)
    print(model.parameters)
    show_filters(0, model.conv1)

    show_images(model, test_loader, 0)

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, valid_loader, optimizer, scheduler, criterion, 8)
    test(model, test_loader, criterion)
