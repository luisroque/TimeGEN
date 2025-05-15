from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.nhits import (
    NHITS,
)


class TimeGENEncoder(nn.Module):
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
        h: int,
        hidden_dims=[256, 128],
        activation: str = "ReLU",
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size
        self.h = h

        activ = getattr(nn, activation)()

        in_features = (
            input_size
            + (input_size + self.h) * futr_input_size
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

        # zero-initialize last layers for stable KL at start
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ):
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


class TimeGEN_S(NHITS):
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
        super(TimeGEN_S, self).__init__(
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

        self.encoder = TimeGENEncoder(
            input_size=input_size,
            latent_dim=latent_dim,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            h=self.h,
        )

        blocks = self.create_stack(
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

        self.latent_to_insample = nn.Linear(latent_dim, input_size)

    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
            z = mu + sigma * eps
        where eps ~ N(0, I).
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu, logvar):
        """
        Numerically safe KL divergence for VAE:
        D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        # We can clamp here too (double safety).
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar_clamped - mu**2 - logvar_clamped.exp(), dim=1)
        )

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
        z_insample = torch.tanh(self.latent_to_insample(z))  # shape [B, L]

        insample_y_cond = insample_y + z_insample  # shape [B, L]
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
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        sum_of_backcasts = initial_flip - residuals
        backcast_reconstruction = sum_of_backcasts.flip(dims=(-1,))

        backcast_reconstruction = self.loss.domain_map(backcast_reconstruction)
        forecast = self.loss.domain_map(forecast)

        return backcast_reconstruction, forecast, mu, logvar

    def training_step(self, batch, batch_idx):
        """
        Custom training step overriding the parent.
        Compute reconstruction loss, forecast loss, KL, then combine.
        """
        windows = self._create_windows(batch, step="train")
        y_idx = batch["y_idx"]
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

        backcast, forecast, mu, logvar = self(windows_batch)

        # print(f"[DEBUG] insample_y: {insample_y.shape}", flush=True)
        # print(f"[DEBUG] backcast: {backcast.shape}", flush=True)
        # print(f"[DEBUG] insample_mask: {insample_mask.shape}", flush=True)
        # print(f"[DEBUG] forecast: {forecast.shape}", flush=True)
        # print(f"[DEBUG] outsample_y: {outsample_y.shape}", flush=True)
        # print(f"[DEBUG] outsample_mask: {outsample_mask.shape}", flush=True)

        recon_loss_fn = MAE(
            horizon_weight=torch.ones(insample_y.shape[-1], device=insample_y.device)
        )
        recon_loss = recon_loss_fn(y_hat=backcast, y=insample_y, mask=insample_mask)
        forecast_loss = self.loss(y_hat=forecast, y=outsample_y, mask=outsample_mask)

        # KL warmup
        # ramp over first 500 steps
        current_ratio = min(1.0, float(self.global_step) / 500.0)
        kl_w = self.final_kl_weight * current_ratio

        kl = self.kl_divergence(mu, logvar)

        recon_weight = 1
        pred_weight = 1

        total_loss = recon_weight * recon_loss + pred_weight * forecast_loss + kl_w * kl

        self.log("train_recon_loss", recon_loss.detach().cpu().item())
        self.log("train_forecast_loss", forecast_loss.detach().cpu().item())
        self.log("train_kl", kl.detach().cpu().item())
        self.log("train_total_loss", total_loss.detach().cpu().item())
        self.log("train_loss", total_loss.detach().cpu().item(), prog_bar=True)

        self.log("latent/mu_mean", mu.mean().detach().cpu().item(), on_step=True)
        self.log(
            "latent/logvar_mean", logvar.mean().detach().cpu().item(), on_step=True
        )

        if self.global_step % 200 == 0:
            print(
                f"[Step {self.global_step}] recon: {recon_loss.item():.4f}, "
                f"forecast: {forecast_loss.item():.4f}, kl: {kl.item():.4f}, "
                f"total: {total_loss.item():.4f}"
            )

        self.train_trajectories.append((self.global_step, total_loss.detach().item()))
        return total_loss

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
            # Create and normalize windows [Ws, L+H, C]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="val", w_idxs=w_idxs)
            original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
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

            # Model Predictions
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
            # Create and normalize windows [Ws, L+H, C]
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

            # Model Predictions
            backcast, forecast, mu, logvar = self(windows_batch)
            # Inverse normalization and sampling
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
