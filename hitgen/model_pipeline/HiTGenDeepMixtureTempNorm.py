from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from hitgen.model_pipeline.HiTGenDeepMixture import HiTGenDeepMixture, HiTGenEncoder
from neuralforecast.losses.pytorch import MAE


class HiTGenDeepMixtureTempNorm(HiTGenDeepMixture):

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
        # NBEATS specific
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        stack_types: list = ["identity", "trend", "seasonality"],
        n_beats_nblocks_stack_1: int = 1,
        n_beats_nblocks_stack_2: int = 1,
        n_beats_nblocks_stack_3: int = 1,
        # general
        nblocks_stack: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        # NHITS specific
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
        super(HiTGenDeepMixture, self).__init__(
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

        self.nblocks_stack = nblocks_stack
        self.nbeats_nblocks_stack = [
            n_beats_nblocks_stack_1,
            n_beats_nblocks_stack_2,
            n_beats_nblocks_stack_3,
        ]

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

        blocks = self._create_universal_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            futr_input_size=self.futr_exog_size,
            hist_input_size=self.hist_exog_size,
            stat_input_size=self.stat_exog_size,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            n_harmonics=n_harmonics,
            n_polynomials=n_polynomials,
            latent_dim=latent_dim,
        )
        self.blocks = nn.ModuleList(blocks)

        self.z_proj = nn.Sequential(nn.Linear(latent_dim, 64), nn.Tanh())

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

        # Following RevIN and HINT approach to inverse scaling before the loss
        backcast_raw, _, _ = self._inv_normalization(
            y_hat=backcast, temporal_cols=batch["temporal_cols"], y_idx=y_idx
        )
        forecast_raw, _, _ = self._inv_normalization(
            y_hat=forecast, temporal_cols=batch["temporal_cols"], y_idx=y_idx
        )

        # also put the targets back
        insample_raw, _, _ = self._inv_normalization(
            y_hat=insample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
        )
        outsample_raw, _, _ = self._inv_normalization(
            y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
        )

        recon_loss_fn = MAE(
            horizon_weight=torch.ones(insample_raw.shape[-1], device=insample_y.device)
        )
        recon_loss = recon_loss_fn(
            y_hat=backcast_raw, y=insample_raw, mask=insample_mask
        )
        forecast_loss = self.loss(
            y_hat=forecast_raw, y=outsample_raw, mask=outsample_mask
        )

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
