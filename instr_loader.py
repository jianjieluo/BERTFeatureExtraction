import os
import json
import string
import re
import _pickle as cPickle
import numpy as np

import torch
from torch.utils.data import Dataset

def _load_dataset(input_file, preprocessor):
    entries = []
    annotation = json.load(open(input_file, 'r'))

    for item in annotation:
        path_id = item['path_id']
        for i, instr in enumerate(item['instructions']):
            instr_id = str(path_id) + '_' + str(i)
            instr = preprocessor.preprocess(instr)

            entries.append({'instr_id': instr_id, 'instr': instr})

    return entries


class R2RPreprocessor(object):
    """ Class to tokenize and encode a sentence in R2R Task """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, max_seq_length=80):
        self.remove_punctuation = remove_punctuation
        self.max_seq_length = max_seq_length

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def preprocess(self, sentence):
        splited = self.split_sentence(sentence)
        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]
        return ' '.join(splited[:self.max_seq_length])


class InstrLoader(Dataset):
    """Pytorch instruction text loader."""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.PAD_ID = tokenizer.vocab["[PAD]"]

        input_file = self.args.input
        self.dataset_name = os.path.basename(input_file).split('.')[0]
        self.output_dir = os.path.join(self.args.out_root, self.dataset_name) 
        cache_root = 'data/cache'
        cache_path = os.path.join(cache_root, self.dataset_name + '.pkl')

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(cache_path):
            ps = R2RPreprocessor(
                remove_punctuation=(self.args.remove_punctuation == 1),
                max_seq_length=self.args.r2r_max_cap_length
            )
            self.entries = _load_dataset(input_file, ps)

            if self.args.debug:
                self.entries = self.entries[:100]

            self.tokenize_entries()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            self.entries = cPickle.load(open(cache_path, "rb"))

        # update max_seq_length
        self.max_seq_length = max([int(len(x['input_seq'])) for x in self.entries])

    def tokenize_entries(self):
        for entry in self.entries:
            input_seq = self.tokenizer.encode(entry['instr'], add_special_tokens=True)
            entry['input_seq'] = input_seq
            entry.pop('instr')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        instr_id = entry['instr_id']
        input_seq = entry['input_seq']
        input_len = len(input_seq)

        # padding to max_seq_length
        padded_seq = input_seq + [self.PAD_ID] * (self.max_seq_length - input_len)
        attn_mask = [1] * input_len + [0] * (self.max_seq_length - input_len)
        output_path = os.path.join(self.output_dir, instr_id)

        padded_seq = np.array(padded_seq)
        attn_mask = np.array(attn_mask)
            
        return {'input_ids': padded_seq, 'attn_mask': attn_mask, 'output_path': output_path}
