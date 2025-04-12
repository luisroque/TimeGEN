from typing import Tuple, Optional

import numpy as np
import torch.nn as nn

from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.nhits import NHITSBlock, _IdentityBasis
from neuralforecast.models.nbeats import (
    NBEATSBlock,
    SeasonalityBasis,
    TrendBasis,
    IdentityBasis,
)
from hitgen.model_pipeline.HiTGen import HiTGen, HiTGenEncoder


class HiTGenMixture(HiTGen):
    """
    A variant of HiTGen that lets you specify how many blocks are N-HiTS vs. N-BEATS
    based on a tunable param.
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
        # NBEATS specific
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        stack_types: list = ["identity", "trend", "seasonality"],
        n_beats_nblocks_stack_1: int = 3,
        n_beats_nblocks_stack_2: int = 3,
        n_beats_nblocks_stack_3: int = 3,
        # general
        nblocks_stack: list = [3, 3, 3],
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

        blocks = self.create_universal_stack(
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
        )
        self.blocks = nn.ModuleList(blocks)

        self.latent_to_insample = nn.Linear(latent_dim, input_size)

    def create_universal_stack(
        self,
        h,
        input_size,
        futr_input_size,
        hist_input_size,
        stat_input_size,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        stack_types,
        n_polynomials,
        n_harmonics,
    ):
        """
        Dynamically builds a stack of blocks, N-BEATS blocks or N-HiTS blocks.
        """
        blocks = []
        for i in range(len(self.nblocks_stack)):
            total_stack_blocks = self.nblocks_stack[i]
            nbeats_count = self.nbeats_nblocks_stack[i]
            nhits_count = total_stack_blocks - nbeats_count

            basis_type_nbeats = stack_types[i]

            for n_beats_id in range(nbeats_count):
                if basis_type_nbeats == "seasonality":
                    n_theta = (
                        2
                        * (self.loss.outputsize_multiplier + 1)
                        * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                    )
                    basis = SeasonalityBasis(
                        harmonics=n_harmonics,
                        backcast_size=input_size,
                        forecast_size=h,
                        out_features=self.loss.outputsize_multiplier,
                    )

                elif basis_type_nbeats == "trend":
                    n_theta = (self.loss.outputsize_multiplier + 1) * (
                        n_polynomials + 1
                    )
                    basis = TrendBasis(
                        degree_of_polynomial=n_polynomials,
                        backcast_size=input_size,
                        forecast_size=h,
                        out_features=self.loss.outputsize_multiplier,
                    )

                elif basis_type_nbeats == "identity":
                    n_theta = input_size + self.loss.outputsize_multiplier * h
                    basis = IdentityBasis(
                        backcast_size=input_size,
                        forecast_size=h,
                        out_features=self.loss.outputsize_multiplier,
                    )

                else:
                    raise ValueError(f"Unknown NBEATS block type {basis_type_nbeats}")

                nbeats_block = NBEATSBlock(
                    input_size=input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )
                blocks.append(nbeats_block)

            for _ in range(nhits_count):
                freq_down = max(h // n_freq_downsample[i], 1)
                n_theta = input_size + self.loss.outputsize_multiplier * freq_down

                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=self.loss.outputsize_multiplier,
                    interpolation_mode=interpolation_mode,
                )

                nhits_block = NHITSBlock(
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

                blocks.append(nhits_block)

        return blocks
