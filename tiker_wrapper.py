import os
import matplotlib.pyplot as plt
import yfinance as yf
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from learner import StockLearner
from data_loader import StockDataSet, get_dataloader
from helper import drop_c, get_plt_multi_data


class Tiker(object):
    def __init__(self, tikers, usedays=120, interval='1d', start="2015-01-01", smoothing=None,
                 end=None, split=[0.8, 0.2], test_se=None, more_val=True, drop_last=True,
                 bs=200, drop=['Adj Close'], keep=None, weightpath='weights'):
        """
        
        Arguments:
            object {[type]} -- [description]
            tiker {[type]} -- [description]
        
        Keyword Arguments:
            interval {str} -- 1h, 90m, 1d (default: {'1d'})
            start {str} -- start date (default: {"2015-01-01"})
            end {str} -- end date, one day beyond today if None. (default: {None})
            split {list} -- split to train val (default: {[0.8,0.2]})
            test_se {list of str} -- start and end date of test data, no if None. (default: {None})
            more_val {int} -- more val data duplicate from the last of train, because we use previous days to predict first val day. if None just usedays.
        """
        self.tikers = tikers
        self.interval = interval
        self.start = start
        self.test_se = test_se
        self.end = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d') if end is None else end
        self.download()
        self.usedays = usedays
        self.smoothing = smoothing
        # self.more_val = more_val
        self.split_data(split, usedays, more_val)
        self.get_dataloader(usedays, bs, drop=drop, keep=keep, drop_last=drop_last, smoothing=smoothing)
        if not (os.path.exists(weightpath) and os.path.isdir(weightpath)):
            os.mkdir(weightpath)
        self.weightpath = weightpath

    def auto(self, epochs=70, lr=0.0001, predays=5, retrain=False, dropout=0.5,
             dims=[[64], 2048, [1024, 512]], act='relu', droplayer='normal', add_bn=False, 
             dim_feedforward=2048, num_layers=1, nhead=8, transformer=False, shift=0,
             **kw):
        self.set_learner(dropout=dropout, encoder=dims[0], hidden_dim=dims[1], transformer=transformer,
                         decoder=dims[2], droplayer=droplayer, act=act, add_bn=add_bn,
                         dim_feedforward=dim_feedforward, num_layers=num_layers, nhead=nhead)
        if retrain or not self.load(): 
            self.train(epochs, lr, **kw)
            self.load()
        self.plt_predict()
        self.plt_multi_predict(predays, shift=shift)
        
    def download(self):
        self.all_raw_pd = yf.download(self.tikers, start=self.start, end=self.end, interval=self.interval)
        if self.test_se is not None:
            self.download_test()

    def download_test(self):
        if self.test_se is not None:
            self.test_raw_pd = yf.download(self.tikers, start=self.test_se[0], end=self.test_se[1], interval=self.interval)
        else:
            print('No start end date specified for test data')

    def split_data(self, split, usedays, more_val=True):
        self.split = split
        self.usedays = usedays
        self.more_val = more_val
        # self.more_val = usedays if more_val is None else more_val
        
        self.num_data = self.all_raw_pd.shape[0]
        self.num_train = int(self.num_data * self.split[0])
        self.num_val = self.num_data - self.num_train

        if self.more_val: self.num_val += usedays

        self.train_raw_pd = self.all_raw_pd[:self.num_train]
        self.val_raw_pd = self.all_raw_pd[-self.num_val:]

    def get_dataloader(self, usedays, bs, drop=['Adj Close'], keep=None, **kw):
        if self.more_val and self.usedays != usedays: 
            self.split(self.split, usedays)
            # self.usedays = usedays
        self.bs = bs
        self.keep = drop_c(drop, self.all_raw_pd) if keep is None else keep

        self.train_dataloader, self.val_dataloader, self.in_dim, self.normf, self.backf = get_dataloader(self.train_raw_pd,
                                                                                                         self.val_raw_pd, None,
                                                                                                         bs=bs,
                                                                                                         usedays=usedays, keep=self.keep, 
                                                                                                         normmethod='mean_std', **kw)
        if self.test_se is not None:
            self.get_test_dataloader()

    def get_test_dataloader(self, bs=None):
        if self.test_se is not None:
            if bs is None:
                bs = self.bs
            test_dataset = StockDataSet(self.test_raw_pd, usedays=self.usedays, normf=self.normf, keep=self.keep, normmethod='mean_std') 
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
        else:
            print('No start end date specified for test data')

    def set_learner(self, encoder=[64], hidden_dim=2048, decoder=[1024, 512], dropout=0.5, num_layers=1, transformer=False, **kw):
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.decoder = decoder
        self.dropout = dropout
        self.num_layers = num_layers
        self.transformer = transformer
        self.model = 'transoformer' if self.transformer else 'lstm'
    
        self.learner = StockLearner(in_dim=self.in_dim, hidden_dim=hidden_dim, seq_len=self.usedays, decoder=decoder, encoder=encoder,
                                    output_dim=len(self.keep), num_layers=num_layers, dropout=dropout, transformer=transformer, **kw)
    
    def train(self, epochs, lr, save='auto', warmup=0.1, **kw):
        if save == 'auto': 
            self.label = '_'.join(self.tikers.replace('^', '+').split() + [self.model] + list(map(str, [self.interval] + self.encoder + 
                                  [self.hidden_dim] + self.decoder + [self.dropout, self.num_layers, self.smoothing])))
            save = self.label + '.pytorch_weight'
        else:
            save = save
        save = os.path.join(self.weightpath, save)
        self.learner.train(epochs, self.train_dataloader, self.val_dataloader, lr, save=save, warmup=warmup, **kw)
    
    def load(self, path='auto'):
        if path == 'auto':
            self.label = '_'.join(self.tikers.replace('^', '+').split() + [self.model] + list(map(str, [self.interval] + self.encoder + 
                                  [self.hidden_dim] + self.decoder + [self.dropout, self.num_layers, self.smoothing])))
            path = self.label + '.pytorch_weight'
        else:
            path = path
        path = os.path.join(self.weightpath, path)
        if os.path.exists(path): 
            self.learner.load_model(path)
            print(path + ' loaded successfully!')
            return True
        else:
            print('Cannot load the weight')
            print(path + ' does not exist!')
            return False

    def plt_predict(self, dataloader=None, dpi=None):
        pred, true, _ = self.learner.predict(self.val_dataloader if dataloader is None else dataloader)
        pred = self.backf(pred)
        true = self.backf(true)
        for i in range(pred.shape[1]):
            if dpi is not None: plt.figure(dpi=dpi)
            plt.plot(pred[:, i], label='Pred')
            plt.plot(true[:, i], label='True')
            plt.legend()
            plt.title(self.tikers + ' ' + self.keep[i])
            plt.show()
        
    def plt_multi_predict(self, predays, data_raw_pd=None, dpi=150, shift=0):
        if data_raw_pd is None:
            data_raw_pd = self.val_raw_pd[self.keep]
        else:
            data_raw_pd = data_raw_pd[self.keep]
        pred, days = self.learner.predict_multi_multi(data_raw_pd.to_numpy(), predays, self.usedays, self.normf, self.backf, shift=shift)
        pd, pp, pt = get_plt_multi_data(days, pred, data_raw_pd.to_numpy())
        # print()
        for i in range(pp.shape[1]):
            if dpi is not None: plt.figure(dpi=dpi)
            plt.plot(pd, pp[:, i], label='Pred')
            plt.plot(pd, pt[:, i], label='True')
            plt.legend()
            plt.title(self.tikers + ' ' + self.keep[i])
            # plt.xlim(400,630)
            plt.show()

    def plt_test(self, dpi=None):
        self.plt_predict(self.test_dataloader, dpi=dpi)

    def plt_test_multi(self, predays, dpi=150):
        self.plt_multi_predict(predays, self.test_raw_pd, dpi=dpi)



    

