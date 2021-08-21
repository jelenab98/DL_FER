from torch.utils.data import DataLoader
from models import *
from utils import *


TRAIN_PATH = Path("./train.csv")
VALID_PATH = Path("./valid.csv")
TEST_PATH = Path("./test.csv")
EMBEDDINGS = Path("./glove.txt")

ENCODERS = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
ENCODER = None

SEED = 8008135

GLOVE_DIM = 300
FREEZE = False
HIDDEN_DIM = 300
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.45

LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 32
GRADIENT_CLIP = 0.25

VOCAB_MAX_SIZE = -1
VOCAB_MIN_FREQ = 1

SAVE_ROOT = Path("./saves/")
SAVE_PATH = SAVE_ROOT / f"{ENCODER}_{SEED}_{HIDDEN_DIM}_{BIDIRECTIONAL}_{DROPOUT}_-1"


def check_data():
    frequencies_text = get_frequencies(TRAIN_PATH, is_target=False)
    frequencies_label = get_frequencies(TRAIN_PATH, is_target=True)

    vocab_text = Vocab(frequencies_text, max_size=VOCAB_MAX_SIZE, min_freq=VOCAB_MIN_FREQ, is_target=False)
    vocab_label = Vocab(frequencies_label, max_size=VOCAB_MAX_SIZE, min_freq=VOCAB_MIN_FREQ, is_target=True)

    train_dataset = NLPDataset(vocab_text, vocab_label, TRAIN_PATH)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)

    texts, labels, lengths = next(iter(train_loader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")


def dump_properties():
    properties = ("SEED", "ENCODER", "GLOVE", "HIDDEN", "LAYERS", "BIDIRECT", "DROPOUT", "LR", "EPOCHS", "BATCH",
                  "GRADIENT", "VOCAB-MAX-SIZE", "VOCAB-MIN-FREQ")
    variables = (SEED, ENCODER, GLOVE_DIM, HIDDEN_DIM, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, LR, EPOCHS, BATCH_SIZE,
                 GRADIENT_CLIP, VOCAB_MAX_SIZE, VOCAB_MIN_FREQ)
    with (SAVE_PATH/"properties.txt").open(mode="w") as f:
        for name, var in zip(properties, variables):
            f.write(f"{name} -> {var}\n")


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    dump_properties()

    # check_data()

    frequencies_text = get_frequencies(TRAIN_PATH, is_target=False)
    frequencies_label = get_frequencies(TRAIN_PATH, is_target=True)

    vocab_text = Vocab(frequencies_text, max_size=VOCAB_MAX_SIZE, min_freq=VOCAB_MIN_FREQ, is_target=False)
    vocab_label = Vocab(frequencies_label, max_size=VOCAB_MAX_SIZE, min_freq=VOCAB_MIN_FREQ, is_target=True)

    train_dataset = NLPDataset(vocab_text, vocab_label, TRAIN_PATH)
    valid_dataset = NLPDataset(vocab_text, vocab_label, VALID_PATH)
    test_dataset = NLPDataset(vocab_text, vocab_label, TEST_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=pad_collate_fn)

    embeddings = get_embedding_matrix(path=EMBEDDINGS, vocab=vocab_text, dim=300, freeze=FREEZE)

    if ENCODER is None:
        model = Baseline(embeddings).to(device)
    else:
        encoder = ENCODERS[ENCODER](input_size=GLOVE_DIM, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                    dropout=DROPOUT, batch_first=False)
        model = RNNModel(embeddings, encoder, HIDDEN_DIM).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_logger = Logger(SAVE_PATH / "train_log.txt", "Loss, Acc, F1")
    valid_logger = Logger(SAVE_PATH / "valid_log.txt", "Loss, Acc, F1")
    test_logger = Logger(SAVE_PATH / "test_log.txt", "Loss, Acc, F1")

    train_valid(model, train_loader, valid_loader, optimizer, criterion, train_logger,
                valid_logger, SAVE_PATH, EPOCHS, GRADIENT_CLIP)

    model = torch.load(SAVE_PATH/"best_model.pth")
    model = model.to(device)

    acc, p, r, f1, loss = evaluate(model, test_loader, criterion)
    test_stats = f"Loss: {loss:.4f}, Acc: {100 * acc:.2f}%, F1: {100 * f1:.2f}%"
    test_stats2 = f"{loss}, {acc}, {f1}"
    print("[TEST STATS:] " + test_stats)
    test_logger.update(test_stats2)
