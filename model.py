import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, frame_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, 4, dim=1)

        i_t = torch.sigmoid(i_conv + self.W_ci * C_prev)
        f_t = torch.sigmoid(f_conv + self.W_cf * C_prev)
        C_t = f_t * C_prev + i_t * (C_conv)
        o_t = torch.sigmoid(o_conv + self.W_co * C_t)
        H_t = o_t * torch.tanh(C_t)
        return H_t, C_t

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, frame_size):
        super().__init__()
        self.out_channels = out_channels
        self.cell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, frame_size)

    def forward(self, X):
        # X: (B, in_channels, T, H, W)
        device = X.device
        B, _, T, H, W = X.size()

        output = torch.zeros(B, self.out_channels, T, H, W, device=device)
        H_t = torch.zeros(B, self.out_channels, H, W, device=device)
        C_t = torch.zeros(B, self.out_channels, H, W, device=device)

        for t in range(T):
            H_t, C_t = self.cell(X[:,:,t], H_t, C_t)
            output[:,:,t] = H_t
        return output

class Seq2Seq(torch.nn.Module):
    def __init__(self, num_channels, hidden_dims, kernel_sizes, frame_size,
                 dropout=0.0):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.layers = torch.nn.ModuleList()

        in_channels = num_channels
        for i in range(self.num_layers):
            out_channels = hidden_dims[i]
            ksize = kernel_sizes[i]
            pad = (ksize[0]//2, ksize[1]//2)
            self.layers.append(
                ConvLSTM(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    padding=pad,
                    frame_size=frame_size
                )
            )
            in_channels = out_channels

        self.conv_out = torch.nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=num_channels,
            kernel_size=(3,3),
            padding=(1,1)
        )
        
        self.dropout_layer = torch.nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        last_frame = out[:, :, -1]
        out = self.conv_out(last_frame)
        if self.dropout_layer is not None:
            out = self.dropout_layer(out)
        return torch.sigmoid(out)
