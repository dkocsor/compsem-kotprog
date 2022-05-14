from prepare_data import SpacyTokenizedTextDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from cbow import CBOWModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)

EMBEDDING_DIM = 30
CONTEXT_SIZE = 4
EPOCH_NUM = 1000

train_ds = SpacyTokenizedTextDataset("./data/datasets/train-articles",
                                         './data/datasets/train-labels-task-si',
                                         create_vocab=False, load_tokenized=True)

torch.autograd.set_detect_anomaly(True)


def create_context(text, context_size):
    ret = []
    for i in range(context_size, len(text) - context_size):
        word_ctx = []
        for j in range(-context_size, context_size + 1):
            if j != 0:
                word_ctx.append(text[i + j])
        ret.append((word_ctx, text[i]))

    return ret


vocab = train_ds.build_vocab()

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}


class ContextDataset(Dataset):
    def __init__(self, text_dataset, word_to_idx, return_tensors=True):
        self.total_context = []
        for text, _ in text_dataset:
            context = create_context(text, CONTEXT_SIZE)
            for c in context:
                if not return_tensors:
                    self.total_context.append(([word_to_idx[word] for word in c[0]], word_to_idx[c[1]]))
                else:
                    self.total_context.append(
                        torch.tensor([word_to_idx[word] for word in c[0]] + [word_to_idx[c[1]]]))

    def __len__(self):
        return len(self.total_context)

    def __getitem__(self, item):
        return self.total_context[item]


ctx_ds = ContextDataset(train_ds, word_to_idx)
# Context_size, Context_size, target indexek
# 4 + 4 + 1 => shape: (9)

train_loader = DataLoader(ctx_ds, batch_size=512, shuffle=True)
# shape: (512, 9)

model = CBOWModel(len(vocab), EMBEDDING_DIM).to(device)
model.load_state_dict(torch.load('data/masodik_proba/model.pt'))
model.train()

loss_fn = nn.NLLLoss()
opt = torch.optim.Adam(model.parameters())


for epoch in range(EPOCH_NUM):
    total_loss = 0.
    i = 0
    for context_and_target in tqdm(train_loader):
        context = context_and_target[:, :-1].to(device)  # (Batch x 2context) elotte, utana context idx
        target = context_and_target[:, -1].to(device)  # (batch) target idx

        # print(model(context).shape)  # (embed_size x vocab_size)  512x22839

        log_probs = model(context)
        loss = loss_fn(log_probs, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

        # if i % 40 == 39:
        #     print(loss)

        i += 1

    print(total_loss / len(train_loader))

    torch.save(model.state_dict(), 'data/model.pt')



