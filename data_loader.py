import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from helper import get_target_id


class StockDataSet(Dataset):
    def __init__(self, raw_data_pd, target=None, usedays=4, predays=1, smoothing=None,
                 keep=None, normmethod='mean_std', normf=None):
        if keep is not None:
            if type(keep) is str:
                keep = [keep]
            raw_data_pd = raw_data_pd[keep]

        self.target_id = get_target_id(raw_data_pd, target)
        self.smoothing = smoothing
        self.raw_data_pd = raw_data_pd.copy(deep=True)
        self.normmethod = normmethod
        self.normalize(normf)

        self.constuct(usedays)
    
    def normalize(self, normf=None):
        self.raw_data_pd.dropna(subset=self.target_id, inplace=True)
        # fill with previous value
        self.raw_data_pd.ffill(inplace=True)
        # drop first rows with nan
        self.raw_data_pd.dropna(inplace=True)
        if self.smoothing is not None:
            self.raw_data_pd = self.raw_data_pd.ewm(span=self.smoothing).mean()

        if normf is None:
            if self.normmethod == 'mean_std':
                self.mean = self.raw_data_pd.mean(axis=0).to_numpy()
                self.std = self.raw_data_pd.std(axis=0).to_numpy()
                meanb = self.raw_data_pd[self.target_id].mean(axis=0).to_numpy()
                stdb = self.raw_data_pd[self.target_id].std(axis=0).to_numpy()
                
                self.normf = lambda x: (x - self.mean) / self.std
                self.backf = lambda x: x * stdb + meanb
                self.raw_data_pd = self.normf(self.raw_data_pd)
            elif self.normmethod == 'min_max':
                self.min = self.raw_data_pd.min(axis=0).to_numpy()
                self.max = self.raw_data_pd.max(axis=0).to_numpy()
                self.diff = self.max - self.min
                self.normf = lambda x: (x - self.min) / self.diff
                self.backf = lambda x: x * self.diff + self.min
                self.raw_data_pd = self.normf(self.raw_data_pd)
            else:
                raise Exception('Error: no such method')

        else:
            self.normf = normf
            self.raw_data_pd = normf(self.raw_data_pd)

    def constuct(self, usedays):
        xx, yy = [], []
        xnp = self.raw_data_pd.to_numpy()
        ynp = self.raw_data_pd[self.target_id].to_numpy()
        for i in range(len(self.raw_data_pd) - usedays - 1):
            x = xnp[i:i + usedays]
            y = ynp[i + usedays]  # .reshape(-1)
            xx.append(torch.Tensor(x))
            yy.append(torch.Tensor(y))
        self.xx = xx
        self.yy = yy

    def __getitem__(self, i):
        return self.xx[i], self.yy[i]

    def __len__(self):
        return len(self.yy)

def get_dataloader(train_raw_pd, val_raw_pd, target, bs=30, usedays=4, shuffle=True, smoothing=None,
                   weightf=None, num_samples=None, drop_last=True, **kw):
    train_dataset = StockDataSet(train_raw_pd, target, usedays=usedays, smoothing=smoothing, **kw)
    val_dataset = StockDataSet(val_raw_pd, target, usedays=usedays, normf=train_dataset.normf, smoothing=smoothing, **kw)

    if weightf is not None:
        if num_samples is None:
            num_samples = len(train_dataset)

        weights = weightf(len(train_dataset))
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    elif shuffle:
        sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, sampler=sampler, drop_last=drop_last)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    return train_dataloader, val_dataloader, train_dataset[0][0][0].size()[0], train_dataset.normf, train_dataset.backf
