import torch
from transformers import RobertaModel, RobertaTokenizerFast
from prepare_data import TextDatasetBertTokenizer2
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Randomize, f1_score_, precision_, recall_
import torch.nn.functional as F

torch.manual_seed(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_SIZE = 512
LSTM_LAYERS = 2
LSTM_HIDDEN = 50
BATCH_SIZE = 10


ds = TextDatasetBertTokenizer2('./data/datasets/train-articles', "./data/datasets/train-labels-task-si",
                               RobertaTokenizerFast.from_pretrained('roberta-large'))

val_ds = TextDatasetBertTokenizer2('./data/datasets/dev-articles', "./data/datasets/dev-labels-task-si",
                                   RobertaTokenizerFast.from_pretrained('roberta-large'))

r = Randomize(ds, SEQ_SIZE)

# r_proba = Randomize(val_ds, SEQ_SIZE)


h_eval, c_eval = torch.randn(LSTM_LAYERS, 1, LSTM_HIDDEN, device=device), \
                 torch.randn(LSTM_LAYERS, 1, LSTM_HIDDEN, device=device)  # hogy ugyanaz legyen mindig


def eval_model(model, val_ds, seq_len, device="cuda"):
    model.eval()
    total_steps = 0
    total_metric1 = 0.
    total_metric2 = 0.
    total_metric3 = 0.
    for original_input_ids, original_labels in tqdm(val_ds):
        text_len = len(original_labels)
        assert len(original_labels) == len(original_input_ids)
        total_cls = torch.tensor([])
        total_labels = torch.tensor([])
        for i in range(0, text_len, seq_len):
            total_steps += 1
            if i + seq_len <= text_len:
                remaining = 0
                input_ids = original_input_ids[i:(i + seq_len)]
                labels = original_labels[i:(i + seq_len)]
                attention_mask = torch.ones(seq_len, dtype=torch.int64)
                token_type_ids = torch.zeros(seq_len, dtype=torch.int64)
            else:
                remaining = text_len - i
                pads_int = torch.zeros(i + seq_len - text_len, dtype=torch.int64)
                attention_mask = torch.concat([torch.ones(text_len - i, dtype=torch.int64), pads_int], 0)
                input_ids = torch.concat([original_input_ids[i:], pads_int], 0)
                labels = torch.concat([original_labels[i:], pads_int], 0)
                token_type_ids = torch.zeros(seq_len, dtype=torch.int64)

            tokenized = {'input_ids': input_ids.unsqueeze(0),
                         'token_type_ids': token_type_ids.unsqueeze(0),
                         'attention_mask': attention_mask.unsqueeze(0),
                         'labels': labels.long().unsqueeze(0)}

            with torch.no_grad():
                train = {k: v.to(device) for k, v in tokenized.items()}
                out = model(train['input_ids'], train['attention_mask'], (h_eval, c_eval))
                preds = F.softmax(out, 2)
                cls = torch.argmax(preds, 2)[0].to('cpu')

            if remaining == 0:
                total_cls = torch.concat([total_cls, cls], 0)
                total_labels = torch.concat([total_labels, labels], 0)
            else:
                total_cls = torch.concat([total_cls, cls[:remaining]], 0)
                total_labels = torch.concat([total_labels, labels[:remaining]], 0)

        total_metric1 += f1_score_(total_cls, total_labels)
        total_metric2 += precision_(total_cls, total_labels)
        total_metric3 += recall_(total_cls, total_labels)

    print('\033[93mF1-score:', total_metric1 / len(val_ds),
          'Precision: ', total_metric2 / len(val_ds), 'Recall: ', total_metric3 / len(val_ds), '\033[0m')


class SiModel1(torch.nn.Module):
    def __init__(self, lstm_dim, lstm_layer):
        super(SiModel1, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-large')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear1 = torch.nn.Linear(1024, 128)
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(128, lstm_dim, lstm_layer, batch_first=True)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(lstm_dim, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, mask, hidden=None):
        bert_out = self.bert(input_ids=input, attention_mask=mask)['last_hidden_state']  # (batch, seq, 1024)
        x = self.relu(self.linear1(bert_out))
        if hidden is None:
            h, c = torch.randn(LSTM_LAYERS, x.shape[0], LSTM_HIDDEN, device=device), \
                   torch.randn(LSTM_LAYERS, x.shape[0], LSTM_HIDDEN, device=device)
            x, _ = self.lstm(x, (h, c))  # (Batch, SEQ, 128)
        else:
            x, _ = self.lstm(x, hidden)
        x = self.linear2(x)  # (batch, seq, 2)
        return x


model = SiModel1(LSTM_HIDDEN, LSTM_LAYERS).to(device)

train_loader = DataLoader(r, shuffle=True, batch_size=BATCH_SIZE)


opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2], device=device))

total_loss = 0.
steps = 0
for epoch in range(100):
    model.train()
    for batch_num, train in enumerate(tqdm(train_loader)):
        train['input_ids'] = train['input_ids'].to(device)
        train['attention_mask'] = train['attention_mask'].to(device)
        train['labels'] = train['labels'].to(device)

        out = model(train['input_ids'], train['attention_mask'])

        opt.zero_grad()
        preds = F.softmax(out, 2)
        loss = loss_fn(preds.permute(0, 2, 1), train['labels'])  # crossentropy loss miatt kell permutalni
        loss.backward()
        opt.step()

        total_loss += loss.item()
        steps += 1

    torch.save(model.state_dict(), 'data/robertaLSTM.pt')

    print(total_loss / steps)

    eval_model(model, val_ds, SEQ_SIZE)






