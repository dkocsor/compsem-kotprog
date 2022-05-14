import torch
from cbow import CBOWModel
from prepare_data import SpacyTokenizedTextDataset
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import f1_score_, precision_, recall_

torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_DIM = 30
EPOCH_NUM = 20
LSTM_LAYERS = 2
LSTM_HIDDEN = 30

train_token_ds = SpacyTokenizedTextDataset("./data/datasets/train-articles",
                                     './data/datasets/train-labels-task-si',
                                     create_vocab=False, load_tokenized=True)

vocab = train_token_ds.build_vocab()

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}


def weighted_binary_cross_entropy(output, target, weights=(1, 3)):
    output = torch.clamp(output, min=1e-5, max=1 - 1e-5)
    loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))

    return torch.neg(torch.mean(loss))


class EmbeddingDataset(Dataset):
    def __init__(self, text_dataset, word_to_idx, embedding_model, embedding_dim, return_tensors=True):
        self.total_embeds = []
        self.labels = text_dataset.labels
        for text, _ in tqdm(text_dataset):
            if return_tensors:
                embeds = torch.empty((len(text), embedding_dim))
                for i, word in enumerate(text):
                    try:
                        idx = word_to_idx[word]
                        embeds[i] = embedding_model.get_embedding_from_idx(torch.tensor(idx))
                    except:
                        # print('missing from vocab:', word)
                        embeds[i] = torch.randn(embedding_dim)

                self.total_embeds.append(embeds)

            else:
                raise NotImplementedError()

        assert len(self.total_embeds) == len(self.labels)

    def __len__(self):
        return len(self.total_embeds)

    def __getitem__(self, item):
        # print(self.total_embeds[item].shape, self.labels[item].shape)
        return self.total_embeds[item], self.labels[item]  # (len x embed, len)


class LstmModel(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, lstm_layer):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, lstm_layer)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(lstm_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)  # bidirectional=True  # (len x hidden_dim), (layer x hidden)
        x = self.tanh(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x, hidden


BATCH_SIZE = 1

cbow_model = CBOWModel(len(vocab), EMBEDDING_DIM)
cbow_model.load_state_dict(torch.load('data/masodik_proba/model.pt'))
cbow_model.eval()


model = LstmModel(EMBEDDING_DIM, LSTM_HIDDEN, LSTM_LAYERS)
model.load_state_dict(torch.load('data/lstmModel.pt'))
model.train()
model = model.to(device)

train_ds = EmbeddingDataset(train_token_ds, word_to_idx, cbow_model, EMBEDDING_DIM)

train_loader = DataLoader(train_ds, shuffle=True)

loss_fn = weighted_binary_cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


val_ds = EmbeddingDataset(
    SpacyTokenizedTextDataset("./data/datasets/dev-articles",
                              './data/datasets/dev-labels-task-si',
                              create_vocab=False, load_tokenized=True, load_file='dev_tokenized'),
    word_to_idx, cbow_model, EMBEDDING_DIM
)

h_eval, c_eval = torch.randn(LSTM_LAYERS, LSTM_HIDDEN, device=device), \
                 torch.randn(LSTM_LAYERS, LSTM_HIDDEN, device=device)  # hogy ugyanaz legyen mindig



def eval_model(val_ds, loss_fn, metric, correct_fn):
    total_loss = 0.
    total_metric = 0.
    i = 0
    print('\033[93meval:\033[0m')
    for embeds, labels in tqdm(val_ds):
        embeds, labels = embeds.to(device), labels.to(device)
        h, c = h_eval, c_eval

        with torch.no_grad():
            raw_out, _ = model(embeds, (h, c))
            raw_out = raw_out.view(-1).to('cpu')
            labels = labels.to('cpu')
            # out = correct_fn(raw_out.view(-1))
            out = torch.zeros_like(labels, dtype=torch.float32)
            out[raw_out > 0.5] = 1
            loss = loss_fn(raw_out, labels).item()
            score = metric(out, labels)
            total_metric += score
            total_loss += loss

        i += 1

    print('\033[93mloss:', total_loss / len(val_ds), ', score:', total_metric / len(val_ds), '\033[0m')

    return total_loss, total_metric


for epoch in range(EPOCH_NUM):
    print('\033[93m', epoch, '.epoch:\033[0m')

    total_loss = 0.
    i = 0

    for embeds, labels in tqdm(train_loader):
        embeds, labels = embeds[0].to(device), labels[0].to(device)  # remove batch dim
        h, c = torch.randn(LSTM_LAYERS, LSTM_HIDDEN, device=device), \
               torch.randn(LSTM_LAYERS, LSTM_HIDDEN, device=device)

        out, _ = model(embeds, (h, c))

        # print(out.shape, embeds.shape, h.shape, c.shape)  # len x 1, len x embed, layers x hidden, layers x hidden

        loss = loss_fn(out.view(-1), labels)

        total_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        i += 1

    torch.save(model.state_dict(), 'data/lstmModel.pt')

    print('\033[93m', total_loss / len(train_loader), '\033[0m')
    eval_model(val_ds=val_ds, loss_fn=loss_fn, metric=f1_score_, correct_fn=None)
    eval_model(val_ds=val_ds, loss_fn=loss_fn, metric=precision_, correct_fn=None)
    eval_model(val_ds=val_ds, loss_fn=loss_fn, metric=recall_, correct_fn=None)




