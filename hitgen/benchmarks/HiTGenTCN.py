import torch.nn as nn
from hitgen.benchmarks.HiTGen import HiTGen


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is shape [B, in_channels, L]
        out = self.conv(x)
        out = self.relu(out)
        return out


class TCNEncoder(nn.Module):
    """
    A small TCN stack as the encoder.
    """

    def __init__(
        self,
        input_size,
        latent_dim,
        channels_tcn=[32, 64],
        kernel_size=3,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        layers = []
        in_ch = 1  # single dimension y(t) as channel=1
        for i, out_ch in enumerate(channels_tcn):
            dilation = 2**i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(out_ch, latent_dim)
        self.fc_logvar = nn.Linear(out_ch, latent_dim)

    def forward(self, insample_y, futr_exog, hist_exog, stat_exog):
        # [B, L] -> [B, 1, L]
        x = insample_y.unsqueeze(1)
        x = self.tcn(x)  # shape [B, out_ch, L_out]
        # can take the last step or global avg
        features = x[:, :, -1]  # [B, out_ch]
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class HiTGenTCN(HiTGen):
    """
    A variant of HiTGen that uses a TCN-based encoder.
    """

    def __init__(self, *args, channels_tcn, kernel_size_tcn, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite the original self.encoder
        self.encoder = TCNEncoder(
            input_size=self.input_size,
            latent_dim=self.encoder.latent_dim,
            channels_tcn=channels_tcn,
            kernel_size=kernel_size_tcn,
        )
