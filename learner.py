import torch
import numpy as np
from model import LSTMStockModel
from helper import warmup_linear

class StockLearner(object):
    def __init__(self, in_dim=0, hidden_dim=128, seq_len=200, encoder=[54], decoder=[1024, 512], state_dict=None, transformer=False, **kw):
        self.model = LSTMStockModel(in_dim=in_dim, hidden_dim=hidden_dim, seq_len=seq_len, encoder=encoder, decoder=decoder, transformer=transformer, **kw)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if state_dict is not None:
            self.load_model(state_dict)
        if self.device.type == 'cuda':
            self.model.cuda()

        self.optimizer = None
        self.lr = 0.001
        self.best_loss = 1e10
        self.val_losses = []
        # self.train_data

    def train(self, epoch, train_data, val_data, lr=0.001, wd=0.01, save=None,
              optimizer='adam', momentum=0.9,
              startsave=20, scheduler='warmup', warmup=0.1, factor=0.2, patience=0.1, verbose=True, min_lr=1e-5,
              base_lr=2e-6, max_lr=5e-4, step_size_up=200, cycle_momentum=False, mode='triangular2'):

        ebatches = len(train_data) // 10
        if ebatches <= 0:
            ebatches = 2
        if self.optimizer is None or lr != self.lr:
            self.lr = lr
            if optimizer == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=lr, weight_decay=wd)
            elif optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=lr, momentum=momentum, nesterov=True)
            elif optimizer == 'rms':
                self.optimizer = torch.optim.RMSprop(
                    self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
            elif optimizer == 'adad':
                self.optimizer = torch.optim.Adadelta(
                    self.model.parameters(), lr=lr, weight_decay=wd)
            else:
                raise Exception('No such optin method')

            if scheduler == 'warmup':
                print('Using warmup scheduler with warmup {}'.format(warmup))
                def lr_lambda(e): return warmup_linear(e, epoch, warmup)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda)
            elif scheduler == 'reduce':
                if type(patience) is float:
                    patience = int(patience * epoch)
                print('Using ReduceLROnPlateau on scheduler with factor={}, patience={}'.format(
                    factor, patience))
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr)
            elif scheduler == 'cycle':
                print('Using cycle lr with base_lr={}, max_lr={}'.format(
                    base_lr, max_lr))
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, cycle_momentum=cycle_momentum, step_size_up=step_size_up)
            elif scheduler is not None:
                raise Exception('No such lr method')
        val_loss = 0
        for i in range(epoch):
            print('** Epoch {} **'.format(i + 1))
            for step, batch in enumerate(train_data):
                # add batch to gpu
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                b_x, b_y = batch

                self.optimizer.zero_grad()

                # forward pass
                loss, _ = self.model(b_x, b_y)
                # backward pass
                loss.backward()
                # update parameters
                self.optimizer.step()
                if step % ebatches == 0:
                    print("Step [{}/{}] train loss: {}".format(step,
                                                               len(train_data), loss.item()))
                if scheduler == 'cycle':
                    self.scheduler.step()
                # self.b_model.zero_grad()
                # pbar.update(1)
            print('------------------------------------------')
            _, _, val_loss = self.predict(val_data)
            if save is not None and val_loss < self.best_loss and i > startsave:
                self.best_loss = val_loss
                self.save_model(save)
            print("Val loss: {}".format(val_loss))
            print('==========================================')
            self.val_losses.append(val_loss)
            if scheduler == 'reduce':
                self.scheduler.step(val_loss)
            elif scheduler == 'warmup':
                self.scheduler.step()

    def predict(self, test_data):
        self.model.eval()
        pred_prices = []
        true_prices = []
        val_loss = 0
        for batch in test_data:
            batch = tuple(t.to(self.device) for t in batch)
            b_x, b_y = batch
            with torch.no_grad():
                loss, price = self.model(b_x, b_y)
            val_loss += loss.item()
            price = price.detach().cpu().numpy()

            pred_prices.extend(price)
            true_prices.extend(b_y.cpu().numpy())
        return np.array(pred_prices), np.array(true_prices), val_loss / len(test_data)

    def predict_multi(self, x, predays):
        self.model.eval()
        x = torch.Tensor([x]).to(self.device)
        pred = []
        with torch.no_grad():
            for i in range(predays):
                p = self.model(x)
                # print(p.size(), x.size())
                x = torch.cat((x[:, 1:], p.unsqueeze(0)), dim=1)
                pred.append(p.clone())

        return torch.cat(pred).cpu().numpy()

    def predict_multi_multi(self, test_data_np, predays, usedays, normf, backf, shift=0):
        test_data = normf(test_data_np)
        # test_data = (test_data_np - mean)/std
        # end = usedays + predays
        end = len(test_data) - (len(test_data) // predays - 1) * predays - shift
        start = end - predays - usedays
        while start < 0:
            start += predays
            end += predays
        preds = []
        # days = [list(range(end-predays, end))]
        days = [list(range(start + usedays))]

        while start + usedays <= len(test_data):
            # print(start+usedays)
            x = test_data[start:start + usedays]
            # print(x.shape, start, start+usedays)
            p = self.predict_multi(x, predays)
            days.append(list(range(end - predays, end)))
            preds.append(backf(p))
            start = start + predays
            end = end + predays

        return preds, days

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
