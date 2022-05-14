import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F


def f1_score_(y_pred, y_true):
    return f1_score(y_true, y_pred, zero_division=0)


def precision_(y_pred, y_true):
    return precision_score(y_true, y_pred, zero_division=0)


def recall_(y_pred, y_true):
    return recall_score(y_true, y_pred, zero_division=0)


class Randomize(Dataset):
    def __init__(self, dataset, seq_len):
        super().__init__()
        self.ds = dataset
        self.seq_len = seq_len
        self.token_type_ids = torch.zeros(self.seq_len, dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        input_ids, label = self.ds[item]
        text_len = input_ids.shape[-1]
        if text_len-self.seq_len > 0:
            r = torch.randint(size=(1,), low=0, high=text_len-self.seq_len)
            input_ids = input_ids[r:(r + self.seq_len)]
            label = label[r:(r + self.seq_len)]
            attention_mask = torch.ones(self.seq_len, dtype=torch.int64)
        else:
            pads = torch.zeros(self.seq_len - text_len)
            pads_int = torch.zeros(self.seq_len - text_len, dtype=torch.int64)
            attention_mask = torch.concat([torch.ones(text_len, dtype=torch.int64), pads_int], 0)
            input_ids = torch.concat([input_ids, pads_int], 0)
            label = torch.concat([label, pads], 0)

        tokenized = {'input_ids': input_ids,
                     'token_type_ids': self.token_type_ids,
                     'attention_mask': attention_mask,
                     'labels': label.long()}

        if tokenized['input_ids'].shape[-1] > self.seq_len:
            print(f"\033[91mAz input_ids ({tokenized['input_ids'].shape[-1]}) "
                  f"hosszabb mint a megadott seq length: {self.seq_len}!\033[00m")

        return tokenized


def eval_model(model, val_ds, seq_len, device="cuda"):
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

        total_metric1 += f1_score_(total_cls, total_labels)
        total_metric2 += precision_(total_cls, total_labels)
        total_metric3 += recall_(total_cls, total_labels)

    print('\033[93mloss:', total_loss / total_steps, ',F1-score:', total_metric1 / len(val_ds),
          'Precision: ', total_metric2 / len(val_ds), 'Recall: ', total_metric3 / len(val_ds), '\033[0m')