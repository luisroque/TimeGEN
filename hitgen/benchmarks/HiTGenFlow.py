import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hitgen.benchmarks.HiTGen import HiTGen


class SimpleRealNVP(nn.Module):
    """
    A minimal RealNVP-like flow block for 1D latent vectors.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_net = nn.Linear(latent_dim, latent_dim)
        self.s_net = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        """
        z -> z' = z * exp(s) + t
        log_det = sum(s)
        """
        s = self.s_net(z)
        t = self.t_net(z)
        z_prime = z * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)  # sum over features
        return z_prime, log_det


class HiTGenFlow(HiTGen):
    """
    Extends the HiTGen VAE with a stack of RealNVP flow layers.
    In the forward pass:
        - sample z ~ q(z|x)
        - pass z through multiple flow layers
        - decode the resulting z_flow to reconstruct + forecast
        - incorporate the log-determinant in your training loss
    """

    def __init__(self, *args, flow_layers=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.flows = nn.ModuleList(
            [SimpleRealNVP(self.encoder.latent_dim) for _ in range(flow_layers)]
        )

    def _reparameterize(self, mu, logvar):
        """
        1) Reparameterize: z_0 = mu + sigma * eps
        2) Pass z_0 through each flow block, accumulating log_det
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_0 = mu + eps * std  # standard Gaussian reparam

        z = z_0
        log_det_sum = 0.0
        for flow in self.flows:
            z, log_det = flow(z)  # shape [B]
            log_det_sum = log_det_sum + log_det.mean()

        return z, log_det_sum

    def forward(self, windows_batch):
        """
        Same as your HiTGen forward, but now the latent z
        comes from the flow-based reparameterization.
        We also return log_det_sum so the training step can use it.
        """
        insample_y = windows_batch["insample_y"]
        insample_mask = windows_batch["insample_mask"]
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        mu, logvar = self.encoder(insample_y, futr_exog, hist_exog, stat_exog)

        z, log_det_sum = self._reparameterize(mu, logvar)

        z_insample = self.latent_to_insample(z)
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
            forecast = forecast + block_forecast

        backcast_reconstruction = insample_y_cond - residuals.flip(dims=(-1,))
        forecast = forecast.squeeze(-1)  # [B, h]

        return backcast_reconstruction, forecast, mu, logvar, log_det_sum

    def training_step(self, batch, batch_idx):
        """
        Override training_step to incorporate log_det in the loss.
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
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        backcast, forecast, mu, logvar, log_det_sum = self.forward(windows_batch)

        recon_loss = F.mse_loss(backcast, insample_y)
        forecast_loss = self.loss(forecast, outsample_y)

        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

        total_loss = recon_loss + forecast_loss + self.kl_weight * kl - log_det_sum

        self.log("train_recon_loss", recon_loss)
        self.log("train_forecast_loss", forecast_loss)
        self.log("train_kl", kl)
        self.log("train_flow_logdet", log_det_sum.mean())
        self.log("train_total_loss", total_loss)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        # TODO: Hack to compute number of windows
        windows = self._create_windows(batch, step="val")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
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

            backcast, forecast, mu, logvar, log_det_sum = self(windows_batch)

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

            backcast, forecast, mu, logvar, log_det_sum = self(windows_batch)
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
