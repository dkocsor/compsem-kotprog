import os
import torch
from torch.utils.data import Dataset
import spacy
import json
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, text_dir, label_dir):
        self.texts = []
        self.labels = []
        for file in os.listdir(text_dir):
            article_name = file.split('.')[0]
            with open(os.path.join(text_dir, article_name + '.txt')) as f_data, \
                    open(os.path.join(label_dir, article_name + '.task-si.labels')) as f_labels:
                self.texts.append(f_data.read())
                span_indices = []
                for line in f_labels:
                    words = line.split()
                    span_indices.append((int(words[1]), int(words[2])))
                self.labels.append(span_indices)

        assert len(self.texts) == len(self.labels)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

    def __len__(self):
        return len(self.texts)


class SpacyTokenizedTextDataset(Dataset):

    def __init__(self, text_dir, label_dir, create_vocab=True, load_tokenized=False, load_file='tokenized'):
        self._create_vocab = create_vocab
        self._nlp = spacy.load('en_core_web_sm')
        self._text_dir = text_dir
        self._label_dir = label_dir
        self.texts = []
        self.labels = []
        if not load_tokenized:
            ds = TextDataset(text_dir, label_dir)
            for raw_text, raw_labels in tqdm(ds):
                token_list, label_tensor = self.__tokenize(raw_text, raw_labels)
                self.texts.append(token_list)
                self.labels.append(label_tensor)

            with open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '_text.json', 'w') as out1, \
                    open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '_label_tensors.json', 'w') as out2:
                json.dump(self.texts, out1)

                json.dump([label_tensor.tolist() for label_tensor in self.labels], out2)
        else:
            with open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '_text.json', 'r') as in1, \
                    open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '_label_tensors.json', 'r') as in2:
                self.texts = json.load(in1)
                self.labels = [torch.tensor(label_list) for label_list in json.load(in2)]

        assert len(self.texts) == len(self.labels)

    def __tokenize(self, raw_text, raw_labels):
        text_tokenized = self._nlp(raw_text)

        token_list = []
        label_list = []
        for token in text_tokenized:
            token_list.append(token.text)
            label_list.append(self.__is_labeled(raw_labels, token.idx))

        return token_list, torch.tensor(label_list)

    @staticmethod
    def __is_labeled(labels, idx):
        for intervals in labels:
            if intervals[0] <= idx < intervals[1]:
                return 1.
        return 0.

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

    def build_vocab(self):

        vocab = set()

        if self._create_vocab:
            for article_no in range(self.__len__()):
                vocab.update(self.__getitem__(article_no)[0])
                with open('/home/daniel/Projects/PYTHON/compsem2/data/vocab.json', 'w') as out:
                    json.dump(list(vocab), out)
        else:
            with open('/home/daniel/Projects/PYTHON/compsem2/data/vocab.json', 'r') as json_file:
                vocab = json.load(json_file)

        return vocab


class TextDatasetBertTokenizer(Dataset):
    def __init__(self, text_dir, label_dir, bert_tokenizer):
        self.dataset = TextDataset(text_dir, label_dir)
        self.tokenizer = bert_tokenizer

    def __getitem__(self, item):
        text, labels = self.dataset[item]
        tokenized = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
        # miden oylan hosszu mint a szoveg
        tokenized['input_ids'] = tokenized['input_ids'][0]
        tokenized['attention_mask'] = tokenized['attention_mask'][0]  # csak 1esek
        tokenized['offset_mapping'] = tokenized['offset_mapping'][0]

        label_tensor = torch.zeros_like(tokenized['input_ids'], dtype=torch.float32)

        if len(labels) == 0:
            return tokenized['input_ids'], label_tensor

        for i, token_range in enumerate(tokenized['offset_mapping']):
            for label in labels:
                if token_range[0] >= label[0] and token_range[1] <= label[1]:
                    label_tensor[i] = 1
                    continue

        return tokenized['input_ids'], label_tensor

    def __len__(self):
        return len(self.dataset)


class TextDatasetBertTokenizer2(Dataset):
    def __init__(self, text_dir, label_dir, bert_tokenizer, load_tokenized=False, load_file='bert_tokenized'):
        self.dataset = TextDataset(text_dir, label_dir)
        self.tokenizer = bert_tokenizer
        self.return_dict = []

        if not load_tokenized:
            for text, labels in tqdm(self.dataset):
                tokenized = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
                tokenized['input_ids'] = tokenized['input_ids'][0]
                tokenized['attention_mask'] = tokenized['attention_mask'][0]  # csak 1esek
                tokenized['offset_mapping'] = tokenized['offset_mapping'][0]

                label_tensor = torch.zeros_like(tokenized['input_ids'], dtype=torch.float32)

                for i, token_range in enumerate(tokenized['offset_mapping']):
                    for label in labels:
                        if token_range[0] >= label[0] and token_range[1] <= label[1]:
                            label_tensor[i] = 1
                            continue

                self.return_dict.append((tokenized['input_ids'], label_tensor))
            # with open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '.json', 'w') as out:
            #     json.dump(self.return_dict, out)
        else:
            raise NotImplementedError()
            with open('/home/daniel/Projects/PYTHON/compsem2/data/' + load_file + '.json', 'r') as in1:
                self.return_dict = json.load(in1)

    def __getitem__(self, item):
        return self.return_dict[item]

    def __len__(self):
        return len(self.dataset)
