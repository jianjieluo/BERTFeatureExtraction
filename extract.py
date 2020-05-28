import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import torch
from torch.utils.data import DataLoader

from instr_loader import InstrLoader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/raw_data/R2R_test.json', help='raw json annotation file')
    parser.add_argument('--out_root', type=str, default='data/features', help='output root dir')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers at data loader')
    parser.add_argument('--batch_size', type=int, default=32, help='instructions batch size')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='bert model name in transformers package')

    parser.add_argument('--r2r_max_cap_length', type=int, default=80, help='max_cap_len in R2R dataset text preprocess')
    parser.add_argument('--remove_punctuation', type=int, default=1, help='flag for whether to remove punctuation in texts')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--windows', action='store_true')

    args = parser.parse_args()
    return args  

class InstrFeatureExtractor(object):
    def __init__(self, args):
        self.args = args
        self.use_gpu = torch.cuda.is_available()
        
        # setup tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)

        # setup model
        model = BertModel.from_pretrained(self.args.bert_model)
        self.model = model.cuda() if self.use_gpu else model

        # setup loader
        dataset = InstrLoader(self.args, self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=self.use_gpu
        )
        self.loader = tqdm(loader, ascii=True)

    def extract_feature(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.loader:
                input_ids = data['input_ids']
                attn_masks = data['attn_mask']
                output_paths = data['output_path']

                if self.args.windows:
                    input_ids = input_ids.long()
                    attn_masks = attn_masks.long()

                if self.use_gpu:
                    input_ids = input_ids.cuda()
                    attn_masks = attn_masks.cuda()

                hidden_repr, _ = self.model(input_ids, attention_mask=attn_masks)

                self.dump_features(hidden_repr, attn_masks, output_paths)
    
    def dump_features(self, model_out, attn_masks, output_paths):
        for i, output_path in enumerate(output_paths):
            mask = attn_masks[i]
            seq_len = int(mask.sum())
            features = model_out[i, :seq_len, :]

            np.savez_compressed(
                output_path, 
                features=features.view(seq_len, -1).cpu().numpy(), 
            )

if __name__ == "__main__":
    args = parse_args()
    pprint(args)

    fe = InstrFeatureExtractor(args)
    fe.extract_feature()