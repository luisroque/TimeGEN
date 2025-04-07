from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_windows import BaseWindows


class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct the backcast (in-sample) and produce the forecast (future).
        """
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros(
                (len(knots), self.forecast_size), device=knots.device
            )
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]
            forecast = forecast[:, None, :]
        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSBlock(nn.Module):
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
    ):
        super().__init__()
        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        total_input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
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

        theta = self.mlp(insample_y)
        backcast, forecast = self.basis(theta)

        return backcast, forecast


class HiTGenEncoder(nn.Module):
    """
    The encoder maps the time-series input (and optionally exogenous features)
    into a latent distribution parameterized by (mu, logvar).
    """

    def __init__(
        self,
        input_size: int,
        latent_dim: int,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        hidden_dims=[256, 128],
        activation: str = "ReLU",
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        activ = getattr(nn, activation)()

        in_features = (
            input_size
            + input_size * futr_input_size
            + input_size * hist_input_size
            + stat_input_size
        )
        modules = []
        prev_dim = in_features
        for hdim in hidden_dims:
            modules.append(nn.Linear(prev_dim, hdim))
            modules.append(activ)
            prev_dim = hdim

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self.net = nn.Sequential(*modules)

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ):
        # flatten or combine all features into one vector
        batch_size = insample_y.size(0)

        insample_y = insample_y.reshape(batch_size, -1)

        if self.futr_input_size > 0:
            futr_exog = futr_exog.reshape(batch_size, -1)
        else:
            futr_exog = torch.zeros(batch_size, 0, device=insample_y.device)

        if self.hist_input_size > 0:
            hist_exog = hist_exog.reshape(batch_size, -1)
        else:
            hist_exog = torch.zeros(batch_size, 0, device=insample_y.device)

        if self.stat_input_size > 0:
            stat_exog = stat_exog.reshape(batch_size, -1)
        else:
            stat_exog = torch.zeros(batch_size, 0, device=insample_y.device)

        x = torch.cat([insample_y, futr_exog, hist_exog, stat_exog], dim=1)

        hidden = self.net(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


def kl_divergence(mu, logvar):
    """
    Standard KL Divergence for VAE:
    D_KL( q(z|x) || p(z) )
      = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))


class HiTGen(BaseWindows):
    """
    A VAE-like model that encodes the time-series input into a latent space,
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
        stack_types: list = ["identity", "identity"],
        n_blocks: list = [1, 1],
        mlp_units: list = 2 * [[256, 256]],
        n_pool_kernel_size: list = [2, 1],
        n_freq_downsample: list = [2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta=0.1,
        activation="ReLU",
        # VAE-specific hyperparams
        latent_dim=64,
        encoder_hidden_dims=[256, 128],
        loss=MAE(),
        valid_loss=None,
        # training parameters
        kl_weight=1e-3,  # weight KL divergence
        max_steps=1000,
        learning_rate=1e-3,
        num_lr_decays=3,
        early_stop_patience_steps=-1,
        val_check_steps=100,
        batch_size=32,
        valid_batch_size=None,
        windows_batch_size=1024,
        inference_windows_batch_size=-1,
        start_padding_enabled=False,
        step_size=1,
        scaler_type="identity",
        random_seed=1,
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

        self.kl_weight = kl_weight
        self.encoder = HiTGenEncoder(
            input_size=input_size,
            latent_dim=latent_dim,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
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
        )
        self.blocks = nn.ModuleList(blocks)

        # a small MLP to map from (latent_dim) -> something that can be appended
        # to the in-sample input

        # To think: we can incorporate z deeper
        # into the blocks later
        self.latent_to_insample = nn.Linear(latent_dim, input_size)

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
    ):
        block_list = []
        for i in range(len(stack_types)):
            for _ in range(n_blocks[i]):
                assert stack_types[i] == "identity", "Only 'identity' is implemented."

                # n_theta = in-sample length + forecast length
                n_theta = input_size + self.loss.outputsize_multiplier * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.loss.outputsize_multiplier,
                    interpolation_mode=interpolation_mode,
                )
                block = NHITSBlock(
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
                )
                block_list.append(block)
        return block_list

    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
            z = mu + sigma * eps
        where eps ~ N(0, I).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]  # [B, L]
        insample_mask = windows_batch["insample_mask"]  # [B, L]
        futr_exog = windows_batch["futr_exog"]  # [B, L+h, F] or [B, L+H, F]
        hist_exog = windows_batch["hist_exog"]  # [B, L, X]
        stat_exog = windows_batch["stat_exog"]  # [B, S]

        mu, logvar = self.encoder(insample_y, futr_exog, hist_exog, stat_exog)
        z = self._reparameterize(mu, logvar)

        # add the latent embedding to the in-sample
        z_insample = self.latent_to_insample(z)  # shape [B, L]
        insample_y_cond = insample_y + z_insample  # shape [B, L]

        residuals = insample_y_cond.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # shape [B, 1, 1]
        forecast = forecast.repeat(1, self.h, 1)  # shape [B, h, 1]

        for block in self.blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            # backcast => [B, L]
            # block_forecast => [B, h, 1]

            residuals = (residuals - backcast) * insample_mask  # [B, L]
            forecast = forecast + block_forecast  # [B, h, 1]

        # backcast reconstruction => shape [B, L]
        backcast_reconstruction = insample_y_cond - residuals.flip(dims=(-1,))

        forecast = forecast.squeeze(-1)  # shape [B, h]

        return backcast_reconstruction, forecast, mu, logvar

    def training_step(self, batch, batch_idx):
        """
        Custom training step overriding the parent.
        Compute reconstruction loss, forecast loss, KL, then combine.
        """
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
        original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)

        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L + h, F]
            hist_exog=hist_exog,  # [Ws, L, X]
            stat_exog=stat_exog,
        )  # [Ws, S]

        backcast, forecast, mu, logvar = self.forward(windows_batch)

        recon_loss = F.mse_loss(backcast, insample_y)

        forecast_loss = self.loss(forecast, outsample_y)

        kl = kl_divergence(mu, logvar)

        total_loss = recon_loss + forecast_loss + self.kl_weight * kl

        self.log("train_recon_loss", recon_loss)
        self.log("train_forecast_loss", forecast_loss)
        self.log("train_kl", kl)
        self.log("train_total_loss", total_loss)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        # TODO: Hack to compute number of windows
        windows = self._create_windows(batch, step="val")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
            # create and normalize windows [Ws, L+H, C]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="val", w_idxs=w_idxs)
            original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
            windows = self._normalization(windows=windows, y_idx=y_idx)

            (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)

            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L + h, F]
                hist_exog=hist_exog,  # [Ws, L, X]
                stat_exog=stat_exog,
            )  # [Ws, S]

            backcast, forecast, mu, logvar = self(windows_batch)

            valid_loss_batch = self._compute_valid_loss(
                outsample_y=original_outsample_y,
                output=forecast,
                outsample_mask=outsample_mask,
                temporal_cols=batch["temporal_cols"],
                y_idx=batch["y_idx"],
            )
            valid_losses.append(valid_loss_batch)
            batch_sizes.append(len(forecast))

        valid_loss = torch.stack(valid_losses)
        batch_sizes = torch.tensor(batch_sizes, device=valid_loss.device)
        batch_size = torch.sum(batch_sizes)
        valid_loss = torch.sum(valid_loss * batch_sizes) / batch_size

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log(
            "valid_loss",
            valid_loss.detach().item(),
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        self.validation_step_outputs.append(valid_loss)
        return valid_loss

    def predict_step(self, batch, batch_idx):

        # TODO: Hack to compute number of windows
        windows = self._create_windows(batch, step="predict")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        y_hats = []
        for i in range(n_batches):
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="predict", w_idxs=w_idxs)
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
                self._parse_windows(batch, windows)
            )

            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L + h, F]
                hist_exog=hist_exog,  # [Ws, L, X]
                stat_exog=stat_exog,
            )  # [Ws, S]

            backcast, forecast, mu, logvar = self(windows_batch)
            if self.loss.is_distribution_output:
                _, y_loc, y_scale = self._inv_normalization(
                    y_hat=torch.empty(
                        size=(insample_y.shape[0], self.h),
                        dtype=forecast[0].dtype,
                        device=forecast[0].device,
                    ),
                    temporal_cols=batch["temporal_cols"],
                    y_idx=y_idx,
                )
                distr_args = self.loss.scale_decouple(
                    output=forecast, loc=y_loc, scale=y_scale
                )
                _, sample_mean, quants = self.loss.sample(distr_args=distr_args)
                y_hat = torch.concat((sample_mean, quants), axis=2)

                if self.loss.return_params:
                    distr_args = torch.stack(distr_args, dim=-1)
                    distr_args = torch.reshape(
                        distr_args, (len(windows["temporal"]), self.h, -1)
                    )
                    y_hat = torch.concat((y_hat, distr_args), axis=2)
            else:
                y_hat, _, _ = self._inv_normalization(
                    y_hat=forecast,
                    temporal_cols=batch["temporal_cols"],
                    y_idx=y_idx,
                )
            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=0)
        return y_hat
