import torch
from transformers import RobertaModel, RobertaTokenizerFast, BertForTokenClassification, BertTokenizerFast, BertConfig
from prepare_data import TextDatasetBertTokenizer, TextDatasetBertTokenizer2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import Randomize, f1_score_, precision_, recall_

torch.manual_seed(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_SIZE = 512


ds = TextDatasetBertTokenizer2('./data/datasets/train-articles', "./data/datasets/train-labels-task-si",
                              BertTokenizerFast.from_pretrained('bert-base-uncased'))

val_ds = TextDatasetBertTokenizer2('./data/datasets/dev-articles', "./data/datasets/dev-labels-task-si",
                              BertTokenizerFast.from_pretrained('bert-base-uncased'))

r = Randomize(ds, SEQ_SIZE)


model = BertForTokenClassification.from_pretrained('bert-base-uncased').to(device)


def eval_model(model, val_ds, seq_len):
    model.eval()
    total_loss = 0.
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
                inputs = {k: v.to(device) for k, v in tokenized.items()}
                out = model(**inputs)
                preds = F.softmax(out.logits, 2)
                cls = torch.argmax(preds, 2)[0].to('cpu')

                loss = out.loss.item()

                total_loss += loss

            if remaining == 0:
                total_cls = torch.concat([total_cls, cls], 0)
                total_labels = torch.concat([total_labels, labels], 0)
            else:
                total_cls = torch.concat([total_cls, cls[:remaining]], 0)
                total_labels = torch.concat([total_labels, labels[:remaining]], 0)

        m1 = f1_score_(total_cls, total_labels)
        m2 = precision_(total_cls, total_labels)
        m3 = recall_(total_cls, total_labels)

        total_metric1 += m1
        total_metric2 += m2
        total_metric3 += m3

    print('\033[93mloss:', total_loss / total_steps, ',F1-score:', total_metric1 / len(val_ds),
          'Precision: ', total_metric2 / len(val_ds), 'Recall: ', total_metric3 / len(val_ds), '\033[0m')


train_loader = DataLoader(r, batch_size=10, shuffle=True)

opt = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 6], device=device))

total_loss = 0.
steps = 0
for epoch in range(10):
    model.train()
    for batch_num, train in enumerate(tqdm(train_loader)):
        train['input_ids'] = train['input_ids'].to(device)
        train['token_type_ids'] = train['token_type_ids'].to(device)
        train['attention_mask'] = train['attention_mask'].to(device)
        train['labels'] = train['labels'].to(device)

        out = model(**train)

        opt.zero_grad()
        preds = F.softmax(out.logits, 2)
        loss = loss_fn(preds.permute(0, 2, 1), train['labels'])  # crossentropy loss miatt kell permutalni
        loss.backward()
        opt.step()

        # print(out.loss.item())

        total_loss += out.loss.item()
        steps += 1

    model.save_pretrained("data/BertForTokenClassification")

    print(total_loss / steps)

    eval_model(model, val_ds, SEQ_SIZE)


eval_model(model, val_ds, SEQ_SIZE)



