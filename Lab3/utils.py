<<<<<<< HEAD
from sklearn.metrics import confusion_matrix as conf_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch.nn import Embedding
from pathlib import Path
from tqdm import tqdm


import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
PADDING_TOKEN = "<PAD>"  # 0
UNKNOWN_TOKEN = "<UNK>"  # 1


class Instance:
    def __init__(self, input_text: [str], target: str):
        self.text = input_text
        self.label = target


class Vocab:
    def __init__(self, frequencies: dict, max_size: int = -1, min_freq: int = 0, is_target: bool = False):
        if is_target:
            self.stoi = dict()
            self.itos = dict()
        else:
            self.stoi = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
            self.itos = {0: PADDING_TOKEN, 1: UNKNOWN_TOKEN}
        self.is_target = is_target
        self.max_size = max_size
        self.min_freq = min_freq

        i = len(self.itos)
        for key, value in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            if (self.max_size != -1) and (len(self.itos) >= self.max_size):
                break
            if value >= self.min_freq:
                self.stoi[key] = i
                self.itos[i] = key
                i += 1
            else:
                break

    def __len__(self):
        return len(self.itos)

    def encode(self, inputs: [str]):
        numericalized_inputs = []
        for token in inputs:
            if token in self.stoi:
                numericalized_inputs.append(self.stoi[token])
            else:
                numericalized_inputs.append(self.stoi[UNKNOWN_TOKEN])

        return torch.tensor(numericalized_inputs)

    def reverse_numericalize(self, inputs: list):
        tokens = []
        for numericalized_item in inputs:
            if numericalized_item in self.itos:
                tokens.append(self.itos[numericalized_item])
            else:
                tokens.append(UNKNOWN_TOKEN)

        return tokens


class NLPDataset(Dataset):
    def __init__(self, text_vocab: Vocab, target_vocab: Vocab, path: Path):
        self.vocab_input_text = text_vocab
        self.vocab_targets = target_vocab
        self.instances = []

        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            text = data[0][i]
            label = data[1][i]
            self.instances.append(Instance(space_tokenizer(text), label.strip()))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        instance_item = self.instances[item]
        text = instance_item.text
        label = [instance_item.label]
        return self.vocab_input_text.encode(text), self.vocab_targets.encode(label)


def space_tokenizer(raw_text: str):
    return raw_text.strip("\n").strip("\r").split(" ")


def get_embedding_matrix(vocab: Vocab, dim: int = 300, freeze: bool = True, path: Path = None):
    matrix = torch.normal(mean=0, std=1, size=(len(vocab), dim))
    matrix[0] = torch.zeros(size=[dim])
    if path is not None:
        data = pd.read_csv(path, header=None, delimiter=" ")
        for i in range(len(data)):
            row = data.loc[i]
            token = row.loc[0]
            if token in vocab.stoi:
                tmp_array = []
                for j in range(1, len(row)):
                    tmp_array.append(row[j])
                matrix[vocab.stoi[token]] = torch.tensor(tmp_array)
    return Embedding.from_pretrained(matrix, padding_idx=0, freeze=freeze)


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    return pad_sequence(texts, batch_first=True, padding_value=pad_index), torch.tensor(labels), lengths


def get_frequencies(path, is_target=False):
    frequencies = {}
    data = pd.read_csv(path, header=None)
    idx = 1 if is_target else 0
    for i in range(len(data)):
        inputs = data[idx][i].strip().split(" ")
        for token in inputs:
            if token in frequencies:
                frequencies[token] += 1
            else:
                frequencies[token] = 1
    return frequencies


def train_valid(model, train_data, valid_data, optimizer, criterion, train_logger,
                valid_logger, save_path: Path = None, epochs=100, gradient_clip=0.25):
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        confusion_matrix = np.zeros(shape=(2, 2))
        losses = []
        for idx, batch in tqdm(enumerate(train_data), total=len(train_data)):
            model.zero_grad()
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x).reshape(y.shape)
            loss = criterion(output, y.float())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            predictions = torch.sigmoid(output).round().int().detach().cpu().numpy()
            confusion_matrix += conf_matrix(y.detach().cpu().numpy(), predictions)
            losses.append(loss.item())

        acc, p, r, f1 = calculate_stats(confusion_matrix)
        train_stats = f"Loss: {np.average(losses):.4f}, Acc: {100 * acc:.2f}%, F1: {100 * f1:.2f}%"
        train_stats2 = f"{np.average(losses)}, {acc}, {f1}"
        print("[TRAIN STATS:] " + train_stats)
        train_logger.update(train_stats2)

        acc_v, p_v, r_v, f1_v, loss_v = evaluate(model, valid_data, criterion)
        valid_stats = f"Loss: {np.average(loss_v):.4f}, Acc: {100 * acc_v:.2f}%, F1: {100 * f1_v:.2f}%"
        valid_stats2 = f"{np.average(loss_v)}, {acc_v}, {f1_v}"
        print("[VALID STATS:] " + valid_stats)
        valid_logger.update(valid_stats2)
        if f1_v > best_f1:
            torch.save(model, save_path / "best_model.pth")
            print(f"Best model saved at {epoch} epoch.")


def calculate_stats(confusion_matrix):
    acc = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    p = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
    r = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    f1 = 2 * p * r / (p + r)
    return acc, p, r, f1


def evaluate(model, data, criterion):
    confusion_matrix = np.zeros(shape=(2, 2))
    losses = list()
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data), total=len(data)):
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x).reshape(shape=y.shape)
            loss = criterion(output, y.float())
            losses.append(loss.item())
            predictions = torch.sigmoid(output).round().int().detach().cpu().numpy()
            confusion_matrix += conf_matrix(y.detach().cpu().numpy(), predictions)

    acc, p, r, f1 = calculate_stats(confusion_matrix)
    loss = np.average(losses)
    return acc, p, r, f1, loss


class Logger:
    def __init__(self, path: Path, start_message: str):
        with path.open(mode="w") as f:
            f.write(f"{start_message}\n")
        self.path = path

    def update(self, message):
        with self.path.open(mode="a") as f:
            f.write(f"{message}\n")
=======
from sklearn.metrics import confusion_matrix as conf_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch.nn import Embedding
from pathlib import Path
from tqdm import tqdm


import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
PADDING_TOKEN = "<PAD>"  # 0
UNKNOWN_TOKEN = "<UNK>"  # 1


class Instance:
    def __init__(self, input_text: [str], target: str):
        self.text = input_text
        self.label = target


class Vocab:
    def __init__(self, frequencies: dict, max_size: int = -1, min_freq: int = 0, is_target: bool = False):
        if is_target:
            self.stoi = dict()
            self.itos = dict()
        else:
            self.stoi = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
            self.itos = {0: PADDING_TOKEN, 1: UNKNOWN_TOKEN}
        self.is_target = is_target
        self.max_size = max_size
        self.min_freq = min_freq

        i = len(self.itos)
        for key, value in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            if (self.max_size != -1) and (len(self.itos) >= self.max_size):
                break
            if value >= self.min_freq:
                self.stoi[key] = i
                self.itos[i] = key
                i += 1
            else:
                break

    def __len__(self):
        return len(self.itos)

    def encode(self, inputs: [str]):
        numericalized_inputs = []
        for token in inputs:
            if token in self.stoi:
                numericalized_inputs.append(self.stoi[token])
            else:
                numericalized_inputs.append(self.stoi[UNKNOWN_TOKEN])

        return torch.tensor(numericalized_inputs)

    def reverse_numericalize(self, inputs: list):
        tokens = []
        for numericalized_item in inputs:
            if numericalized_item in self.itos:
                tokens.append(self.itos[numericalized_item])
            else:
                tokens.append(UNKNOWN_TOKEN)

        return tokens


class NLPDataset(Dataset):
    def __init__(self, text_vocab: Vocab, target_vocab: Vocab, path: Path):
        self.vocab_input_text = text_vocab
        self.vocab_targets = target_vocab
        self.instances = []

        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            text = data[0][i]
            label = data[1][i]
            self.instances.append(Instance(space_tokenizer(text), label.strip()))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        instance_item = self.instances[item]
        text = instance_item.text
        label = [instance_item.label]
        return self.vocab_input_text.encode(text), self.vocab_targets.encode(label)


def space_tokenizer(raw_text: str):
    return raw_text.strip("\n").strip("\r").split(" ")


def get_embedding_matrix(vocab: Vocab, dim: int = 300, freeze: bool = True, path: Path = None):
    matrix = torch.normal(mean=0, std=1, size=(len(vocab), dim))
    matrix[0] = torch.zeros(size=[dim])
    if path is not None:
        data = pd.read_csv(path, header=None, delimiter=" ")
        for i in range(len(data)):
            row = data.loc[i]
            token = row.loc[0]
            if token in vocab.stoi:
                tmp_array = []
                for j in range(1, len(row)):
                    tmp_array.append(row[j])
                matrix[vocab.stoi[token]] = torch.tensor(tmp_array)
    return Embedding.from_pretrained(matrix, padding_idx=0, freeze=freeze)


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    return pad_sequence(texts, batch_first=True, padding_value=pad_index), torch.tensor(labels), lengths


def get_frequencies(path, is_target=False):
    frequencies = {}
    data = pd.read_csv(path, header=None)
    idx = 1 if is_target else 0
    for i in range(len(data)):
        inputs = data[idx][i].strip().split(" ")
        for token in inputs:
            if token in frequencies:
                frequencies[token] += 1
            else:
                frequencies[token] = 1
    return frequencies


def train_valid(model, train_data, valid_data, optimizer, criterion, train_logger,
                valid_logger, save_path: Path = None, epochs=100, gradient_clip=0.25):
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        confusion_matrix = np.zeros(shape=(2, 2))
        losses = []
        for idx, batch in tqdm(enumerate(train_data), total=len(train_data)):
            model.zero_grad()
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x).reshape(y.shape)
            loss = criterion(output, y.float())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            predictions = torch.sigmoid(output).round().int().detach().cpu().numpy()
            confusion_matrix += conf_matrix(y.detach().cpu().numpy(), predictions)
            losses.append(loss.item())

        acc, p, r, f1 = calculate_stats(confusion_matrix)
        train_stats = f"Loss: {np.average(losses):.4f}, Acc: {100 * acc:.2f}%, F1: {100 * f1:.2f}%"
        train_stats2 = f"{np.average(losses)}, {acc}, {f1}"
        print("[TRAIN STATS:] " + train_stats)
        train_logger.update(train_stats2)

        acc_v, p_v, r_v, f1_v, loss_v = evaluate(model, valid_data, criterion)
        valid_stats = f"Loss: {np.average(loss_v):.4f}, Acc: {100 * acc_v:.2f}%, F1: {100 * f1_v:.2f}%"
        valid_stats2 = f"{np.average(loss_v)}, {acc_v}, {f1_v}"
        print("[VALID STATS:] " + valid_stats)
        valid_logger.update(valid_stats2)
        if f1_v > best_f1:
            torch.save(model, save_path / "best_model.pth")
            print(f"Best model saved at {epoch} epoch.")


def calculate_stats(confusion_matrix):
    acc = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    p = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
    r = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    f1 = 2 * p * r / (p + r)
    return acc, p, r, f1


def evaluate(model, data, criterion):
    confusion_matrix = np.zeros(shape=(2, 2))
    losses = list()
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data), total=len(data)):
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x).reshape(shape=y.shape)
            loss = criterion(output, y.float())
            losses.append(loss.item())
            predictions = torch.sigmoid(output).round().int().detach().cpu().numpy()
            confusion_matrix += conf_matrix(y.detach().cpu().numpy(), predictions)

    acc, p, r, f1 = calculate_stats(confusion_matrix)
    loss = np.average(losses)
    return acc, p, r, f1, loss


class Logger:
    def __init__(self, path: Path, start_message: str):
        with path.open(mode="w") as f:
            f.write(f"{start_message}\n")
        self.path = path

    def update(self, message):
        with self.path.open(mode="a") as f:
            f.write(f"{message}\n")
>>>>>>> ca8923228a32a1117eff983cbec160e90b72ca02
