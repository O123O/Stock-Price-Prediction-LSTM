import torch
import torch.nn as nn

class LSTMStockModel(nn.Module):
    def __init__(self, in_dim=0, hidden_dim=0, seq_len=0, encoder=[64], decoder=[1024, 512], add_bn=True, nhead=8, dim_feedforward=1024,
                 transformer=False,
                 num_layers=6, output_dim=5, dropout=0.5, act='relu', droplayer='normal'):
        super().__init__()
        self.hidden_dim = hidden_dim

        # encoder
        elist = [in_dim] + encoder
        enc_layers = []
        if add_bn:
            enc_layers.append(nn.BatchNorm1d(seq_len))

        for i in range(len(encoder)):
            enc_layers.append(nn.Linear(elist[i], elist[i + 1]))
            enc_layers.extend([nn.ReLU(),  # nn.BatchNorm1d(seq_len), 
                               nn.Dropout(p=dropout)])

        self.encoder = nn.Sequential(*enc_layers) if len(enc_layers) > 0 else nn.Identity()

        # decoder
        dlist = [hidden_dim] + decoder + [output_dim]
        dec_layers = []
        dec_layers.extend([nn.BatchNorm1d(hidden_dim),  # nn.LayerNorm(hidden_dim), 
                           nn.Dropout(p=dropout)
                           ])
        for i in range(len(dlist) - 1):
            dec_layers.append(nn.Linear(dlist[i], dlist[i + 1]))
            if i < len(dlist) - 2:
                dec_layers.extend([nn.ReLU(),  # nn.BatchNorm1d(dlist[i + 1]), 
                                   nn.Dropout(p=dropout)])

        self.decoder = nn.Sequential(*dec_layers)

        self.transformer = transformer
        if transformer:
            print('Using Transformer')
            self.lstm = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers)
        else:
            print('Using LSTM')
            self.lstm = nn.LSTM(elist[-1], hidden_dim,
                                dropout=dropout, batch_first=True, num_layers=num_layers)

        self.output_dim = output_dim

    def forward(self, hist, y=None):
        out = self.encoder(hist)
        if self.transformer:
            out = self.lstm(torch.transpose(out, 0, 1))
            price = self.decoder(out[-1])
        else:
            _, (out, _) = self.lstm(out)
            price = self.decoder(out[0]) 

        if y is not None:
            lft = nn.MSELoss()
            loss = lft(price, y)
            return loss, price
        else:
            return price
