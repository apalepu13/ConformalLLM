from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch
from torch.utils.data import RandomSampler
import numpy as np

class MedQA_dataset(Dataset):
    """MedQA dataset"""

    def __init__(self, config=None, subset='all'):
        """
        Args: config, handles all details.
        """
        if config is None:
            with open('../../configs/config.json', "r") as f:
                config = json.load(f)
        self.letters = ['A', 'B', 'C', 'D']
        self.medQA_dir = config['filepaths']['medQA_directory']
        self.shuffle_answer_choices = config['shuffle_answers']
        self.questions_dir = self.getDir(subset=subset)
        self.df = self.getMedQA_df()
        print(self.df.head())
        if config['debug']:
            self.df = self.df[::int(1/config['debugFraction'])]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx, :]
        sample['question'] = str(row['question'])
        sample['answer'] = str(row['answer'])
        shuffledInds = np.arange(len(self.letters))
        if self.shuffle_answer_choices:
            np.random.shuffle(shuffledInds)
        letter_dict = {self.letters[i]: shuffledInds[i] for i in range(len(shuffledInds))}
        answer_choices = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        sample['A'] = answer_choices[shuffledInds[0]]
        sample['B'] = answer_choices[shuffledInds[1]]
        sample['C'] = answer_choices[shuffledInds[2]]
        sample['D'] = answer_choices[shuffledInds[3]]
        sample['meta_info'] = str(row['meta_info'])
        sample['answer_idx'] = np.argwhere(shuffledInds == int(letter_dict[row['answer_idx']]))[0]
        return sample

    def getDir(self, subset=None):
        mydir = self.medQA_dir + ('/US/4_options/phrases_no_exclude_')
        if 'calib' in subset:
            mydir = [mydir + 'train.jsonl']
        elif 'val' in subset:
            mydir = [mydir + 'dev.jsonl']
        elif 'test' in subset:
            mydir = [mydir + 'test.jsonl']
        else:
            mydir = [mydir + t for t in ['train.jsonl', 'dev.jsonl', 'test.jsonl']]
        return mydir

    def getMedQA_df(self):
        dfs = []
        for path in self.questions_dir:
            with open(path, 'r') as f:
                r = [json.loads(jline) for jline in f.read().splitlines()]
                for i in range(len(r)):
                    r[i]['A'] = r[i]['options']['A']
                    r[i]['B'] = r[i]['options']['B']
                    r[i]['C'] = r[i]['options']['C']
                    r[i]['D'] = r[i]['options']['D']
                    del r[i]["options"]

            dfs.append(pd.DataFrame(r))

        df = pd.concat(dfs)
        return df

def get_MedQA_loader(config=None, subset='test'):
    MedQA_dat = MedQA_dataset(config, subset)
    return DataLoader(MedQA_dat, batch_size=config['batch_size'],
                      shuffle=False, num_workers=16, prefetch_factor=2, pin_memory=True)