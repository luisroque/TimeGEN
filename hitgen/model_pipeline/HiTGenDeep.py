from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from neuralforecast.losses.pytorch import MAE
from hitgen.model_pipeline.HiTGen import HiTGen, HiTGenEncoder
from neuralforecast.models.nhits import ACTIVATIONS, POOLING, _IdentityBasis


class NHITSBlockLatent(nn.Module):
    """
    This block can function as part of the 'decoder' in the VAE setting.
    It takes a representation (insample_y + exogenous + latent embedding)
    and produces backcast + forecast outputs.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        self.latent_transform = nn.Linear(latent_dim, 64)

        total_input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
            + 64  # extra from latent_z
        )

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()
        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

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

        # output layer => n_theta parameters for the basis
        hidden_layers.append(
            nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)
        )
        self.mlp = nn.Sequential(*hidden_layers)

        self.basis = basis
        self.h = h
        self.input_size = input_size
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
        latent_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        insample_y = insample_y.unsqueeze(1)  # (B,1,L)
        insample_y = self.pooling_layer(insample_y)  # (B,1,pooled_L)
        insample_y = insample_y.squeeze(1)  # (B, pooled_L)

        batch_size = len(insample_y)

        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
            hist_exog = self.pooling_layer(hist_exog)  # (B, C, pooled_L)
            hist_exog = hist_exog.permute(0, 2, 1)  # (B, pooled_L, C)
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # (B, L+H, C) -> (B, C, L+H)
            futr_exog = self.pooling_layer(futr_exog)  # (B, C, pooled_L+pooled_H)
            futr_exog = futr_exog.permute(0, 2, 1)  # (B, pooled_, C)
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        z_embed = torch.tanh(self.latent_transform(latent_z))  # [B, 64]

        insample_y = torch.cat((insample_y, z_embed), dim=1)

        theta = self.mlp(insample_y)
        backcast, forecast = self.basis(theta)

        return backcast, forecast


class HiTGenDeep(HiTGen):
    """
    A VAE-like data_pipeline that encodes the time-series input into a latent space,
    then decodes it (via NHITS blocks) to reconstruct the in-sample portion (backcast)
    and forecast future values.

    The training should include:
        - Reconstruction loss (compare backcast to in-sample y)
        - Forecast loss (compare forecast to actual future y)
        - KL divergence (force latent distribution to be close to N(0, I))
    """

    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True

    def __init__(
        self,
        h,
        input_size,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta=0.0,
        activation="ReLU",
        # VAE-specific hyperparams
        latent_dim=64,
        encoder_hidden_dims=[64, 32],
        # training params
        kl_weight=0.3,  # weight KL divergence
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader=False,
        alias=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super(HiTGen, self).__init__(
            h=h,
            input_size=input_size,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        self.final_kl_weight = kl_weight
        self.latent_dim = latent_dim

        self.encoder = HiTGenEncoder(
            input_size=input_size,
            latent_dim=latent_dim,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            h=self.h,
        )

        blocks = self._create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            latent_dim=latent_dim,
        )
        self.blocks = nn.ModuleList(blocks)

        # still do a small offset to the in-sample
        self.latent_to_insample = nn.Linear(latent_dim, input_size)

        # zero-init final layer in the encoder for stable KL
        nn.init.zeros_(self.encoder.fc_mu.weight)
        nn.init.zeros_(self.encoder.fc_mu.bias)
        nn.init.zeros_(self.encoder.fc_logvar.weight)
        nn.init.zeros_(self.encoder.fc_logvar.bias)

    def _create_stack(
        self,
        h,
        input_size,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        futr_input_size,
        hist_input_size,
        stat_input_size,
        latent_dim,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + self.loss.outputsize_multiplier * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.loss.outputsize_multiplier,
                    interpolation_mode=interpolation_mode,
                )

                nbeats_block = NHITSBlockLatent(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                    latent_dim=latent_dim,
                )

                block_list.append(nbeats_block)

        return block_list

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]  # [B, L]
        insample_mask = windows_batch["insample_mask"]  # [B, L]
        futr_exog = windows_batch["futr_exog"]  # [B, L+h, F] or [B, L+H, F]
        hist_exog = windows_batch["hist_exog"]  # [B, L, X]
        stat_exog = windows_batch["stat_exog"]  # [B, S]

        mu, logvar = self.encoder(insample_y, futr_exog, hist_exog, stat_exog)
        z = self._reparameterize(mu, logvar)

        # add the latent embedding to the in-sample
        # keep it in [-1, 1]
        # add a small offset to in-sample
        z_insample = 0.1 * torch.tanh(self.latent_to_insample(z))  # [B, L]
        insample_y_cond = insample_y + z_insample

        initial_flip = insample_y_cond.flip(dims=(-1,))

        residuals = insample_y_cond.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # shape [B, 1, 1]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
                latent_z=z,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        sum_of_backcasts = initial_flip - residuals
        backcast_reconstruction = sum_of_backcasts.flip(dims=(-1,))

        backcast_reconstruction = self.loss.domain_map(backcast_reconstruction)
        forecast = self.loss.domain_map(forecast)

        return backcast_reconstruction, forecast, mu, logvar
