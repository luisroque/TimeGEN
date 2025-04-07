import torch.nn as nn
from hitgen.benchmarks.HiTGen import HiTGen


class SimpleAttentionEncoder(nn.Module):
    """
    Replace the MLP in the encoder with a single multi-head attention layer
    to capture temporal dependencies better.
    """

    def __init__(
        self,
        input_size,
        latent_dim,
        num_heads=2,
        hidden_dims=[256],
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.embedding = nn.Linear(1, hidden_dims[0])  # each time step is a token
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dims[0], num_heads=num_heads, batch_first=True
        )

        # then project to mu, logvar
        self.fc_mu = nn.Linear(hidden_dims[0], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[0], latent_dim)

    def forward(self, insample_y, futr_exog, hist_exog, stat_exog):
        # naive embedding for now

        x = insample_y.unsqueeze(-1)  # [B, L, 1]
        x = self.embedding(x)  # [B, L, hidden_dims[0]]

        attn_out, _ = self.attn(x, x, x)

        # we can reduce by taking the last token or maybe mean pool
        # for now last token for simplicity
        features = attn_out[:, -1, :]  # shape [B, hidden]

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class HiTGenAttn(HiTGen):
    """
    Replaces the original MLP-based HiTGenEncoder with an attention-based encoder.
    """

    def __init__(self, *args, num_heads=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SimpleAttentionEncoder(
            input_size=self.input_size,
            latent_dim=self.encoder.latent_dim,
            num_heads=num_heads,
            hidden_dims=[256],
        )
