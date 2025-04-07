import torch
import torch.nn as nn
import numpy as np
from hitgen.benchmarks.HiTGen import HiTGen


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSEncoderBlock(nn.Module):
    """
    NHITS-like block for the ENCODER side:
      - Pools insample data (and exogs),
      - Produces 'theta' via MLP,
      - Interprets part of 'theta' as a backcast (to subtract from residual),
      - Projects 'theta' to an embedding vector.

    This block does *not* produce a forecast (like the decoder).
    """

    def __init__(
        self,
        input_size: int,
        n_theta_backcast: int,
        mlp_units: list,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
        block_embedding_dim: int = 64,
    ):
        super().__init__()
        self.block_embedding_dim = block_embedding_dim

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))

        total_input_size = pooled_hist_size
        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        if hist_input_size > 0:
            total_input_size += hist_input_size * pooled_hist_size
        if futr_input_size > 0:
            pooled_futr_size = int(np.ceil((input_size) / n_pool_kernel_size))
            total_input_size += futr_input_size * pooled_futr_size
        if stat_input_size > 0:
            total_input_size += stat_input_size

        assert activation in ACTIVATIONS, f"{activation} not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        hidden_layers = []
        hidden_layers.append(
            nn.Linear(in_features=total_input_size, out_features=mlp_units[0][0])
        )
        for layer_cfg in mlp_units:
            in_f, out_f = layer_cfg
            hidden_layers.append(nn.Linear(in_f, out_f))
            hidden_layers.append(activ)
            if dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=dropout_prob))

        hidden_layers.append(
            nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta_backcast)
        )
        self.mlp = nn.Sequential(*hidden_layers)

        self.n_theta_backcast = n_theta_backcast
        self.input_size = input_size
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        self.embedding_proj = nn.Linear(n_theta_backcast, block_embedding_dim)

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ):
        """
        Returns:
          backcast => [B, input_size]
          block_emb => [B, block_embedding_dim]
        """

        B = insample_y.shape[0]

        insample_y = insample_y.unsqueeze(1)  # (B,1,L)
        insample_y = self.pooling_layer(insample_y)  # (B,1,pooled_L)
        insample_y = insample_y.squeeze(1)  # (B, pooled_L)

        x = [insample_y]

        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # (B, X, L)
            hist_exog = self.pooling_layer(hist_exog)  # (B, X, pooled_L)
            hist_exog = hist_exog.permute(0, 2, 1)  # (B, pooled_L, X)
            x.append(hist_exog.reshape(B, -1))

        if self.futr_input_size > 0:
            futr_exog = futr_exog[:, : self.input_size, :]  # just the first L steps
            futr_exog = futr_exog.permute(0, 2, 1)  # (B, F, L)
            futr_exog = self.pooling_layer(futr_exog)  # (B, F, pooled_L)
            futr_exog = futr_exog.permute(0, 2, 1)  # (B, pooled_L, F)
            x.append(futr_exog.reshape(B, -1))

        if self.stat_input_size > 0:
            x.append(stat_exog.reshape(B, -1))

        x_cat = torch.cat(x, dim=1)  # shape [B, total_input_size]
        theta = self.mlp(x_cat)  # shape [B, n_theta_backcast]

        backcast = theta[:, : self.input_size]

        block_emb = self.embedding_proj(theta)  # [B, block_embedding_dim]

        return backcast, block_emb


class NHITSEncoder(nn.Module):
    """
    An NHITS-like multi-block ENCODER. Each block is a residual stage:
      - partial backcast,
      - partial embedding,
    Then we combine all partial embeddings to produce a single (mu, logvar).
    """

    def __init__(
        self,
        input_size: int,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        # block hyperparams
        stack_types: list,
        n_blocks: list,
        mlp_units: list,
        n_pool_kernel_size: list,
        activation: str,
        pooling_mode: str = "MaxPool1d",
        dropout_prob_theta: float = 0.1,
        block_embedding_dim: int = 32,
        # final latent dimension
        latent_dim: int = 64,
    ):
        super().__init__()

        self.input_size = input_size
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size
        self.latent_dim = latent_dim
        self.block_embedding_dim = block_embedding_dim

        blocks = []
        for i in range(len(stack_types)):
            assert stack_types[i] == "identity", "Only 'identity' type is implemented."
            for _ in range(n_blocks[i]):
                block = NHITSEncoderBlock(
                    input_size=input_size,
                    n_theta_backcast=input_size,
                    mlp_units=mlp_units,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                    block_embedding_dim=block_embedding_dim,
                )
                blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.n_total_blocks = len(blocks)

        # final aggregator => from [B, #blocks * block_embedding_dim] -> [B, latent_dim]
        self.fc_mu = nn.Linear(self.n_total_blocks * block_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(
            self.n_total_blocks * block_embedding_dim, latent_dim
        )

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ):
        """
        Returns mu, logvar for the VAE reparameterization.
        """
        residual = insample_y.flip(dims=(-1,))
        block_embeddings = []

        for block in self.blocks:
            backcast, emb_i = block(
                insample_y=residual,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residual = residual - backcast  # no mask here

            block_embeddings.append(emb_i)

        z_enc = torch.cat(
            block_embeddings, dim=1
        )  # shape [B, #blocks*block_embedding_dim]

        mu = self.fc_mu(z_enc)
        logvar = self.fc_logvar(z_enc)

        return mu, logvar


class HiTGenHier(HiTGen):
    """
    HiTGen variant that uses an NHITS-like multi-block encoder:
     - Instead of HiTGenEncoder (simple MLP),
     - We build `NHITSEncoder` with multiple blocks, residual approach.
    """

    def __init__(self, *args, **kwargs):
        stack_types = kwargs.get("enc_stack_types", ["identity"])
        n_blocks = kwargs.get("enc_n_blocks", [2])
        mlp_units = kwargs.get("enc_mlp_units", [[256, 256]])
        n_pool_kernel_size = kwargs.get("enc_n_pool_kernel_size", [2])
        activation = kwargs.get("enc_activation", "ReLU")
        pooling_mode = kwargs.get("enc_pooling_mode", "MaxPool1d")
        dropout_prob_theta = kwargs.get("enc_dropout_prob_theta", 0.1)
        block_embedding_dim = kwargs.get("enc_block_embedding_dim", 64)
        latent_dim = kwargs.get("latent_dim", 64)

        super().__init__(*args, **kwargs)

        self.encoder = NHITSEncoder(
            input_size=self.input_size,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            pooling_mode=pooling_mode,
            activation=activation,
            dropout_prob_theta=dropout_prob_theta,
            block_embedding_dim=block_embedding_dim,
            latent_dim=latent_dim,
        )

    def forward(self, windows_batch):
        """
        Standard VAE flow:
          1) Encode => (mu, logvar) using multi-block NHITS encoder
          2) Reparameterize => z
          3) Insert z into in-sample
          4) Use standard NHITS-based decoding from the parent
        """
        insample_y = windows_batch["insample_y"]  # [B, L]
        insample_mask = windows_batch["insample_mask"]  # [B, L]
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        mu, logvar = self.encoder(insample_y, futr_exog, hist_exog, stat_exog)

        z = self._reparameterize(mu, logvar)

        z_insample = self.latent_to_insample(z)  # shape [B, L]
        insample_y_cond = insample_y + z_insample

        residuals = insample_y_cond.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None].repeat(1, self.h, 1)

        for block in self.blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast += block_forecast

        backcast_reconstruction = insample_y_cond - residuals.flip(dims=(-1,))
        forecast = forecast.squeeze(-1)  # [B, h]

        return backcast_reconstruction, forecast, mu, logvar
