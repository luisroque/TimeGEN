from os import cpu_count
import torch

from ray import tune

from hitgen.benchmarks.HiTGen import HiTGen
from hitgen.benchmarks.HiTGenFlow import HiTGenFlow
from hitgen.benchmarks.HiTGenAttn import HiTGenAttn
from hitgen.benchmarks.HiTGenTCN import HiTGenTCN
from hitgen.benchmarks.HiTGenHier import HiTGenHier

from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import BaseAuto


class AutoHiTGen(BaseAuto):

    default_config = {
        "latent_dim": tune.choice([16, 32, 64, 128]),
        "kl_weight": tune.loguniform(1e-5, 1e-1),
        "encoder_hidden_dims": tune.choice([[128, 64], [256, 128], [512, 256]]),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 1],
                [8, 4, 1],
                [16, 8, 1],
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [24, 12, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # trainer parameters
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,  # Will default to MAE if not specified
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        # Define search space, input/output sizes
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGen, self).__init__(
            cls_model=HiTGen,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenFlow(BaseAuto):

    default_config = {
        "latent_dim": tune.choice([16, 32, 64, 128]),
        "kl_weight": tune.loguniform(1e-5, 1e-1),
        "encoder_hidden_dims": tune.choice([[128, 64], [256, 128], [512, 256]]),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 1],
                [8, 4, 1],
                [16, 8, 1],
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [24, 12, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # trainer parameters
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,  # Will default to MAE if not specified
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        # Define search space, input/output sizes
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenFlow, self).__init__(
            cls_model=HiTGenFlow,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenAttn(BaseAuto):
    default_config = {
        "latent_dim": tune.choice([16, 32, 64, 128]),
        "kl_weight": tune.loguniform(1e-5, 1e-1),
        "encoder_hidden_dims": tune.choice([[128, 64], [256, 128], [512, 256]]),
        "num_heads": tune.choice([1, 2, 4]),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 1],
                [8, 4, 1],
                [16, 8, 1],
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [24, 12, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # trainer parameters
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,  # Will default to MAE if not specified
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenAttn, self).__init__(
            cls_model=HiTGenAttn,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config


class AutoHiTGenTCN(BaseAuto):

    default_config = {
        "latent_dim": tune.choice([16, 32, 64]),
        "kl_weight": tune.loguniform(1e-5, 1e-1),
        "encoder_hidden_dims": tune.choice([[128, 64], [256, 128]]),
        "channels_tcn": tune.choice([[32, 64], [64, 64], [64, 128]]),
        "kernel_size_tcn": tune.choice([2, 3, 5]),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 1],
                [8, 4, 1],
                [16, 8, 1],
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [24, 12, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # trainer parameters
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,  # Will default to MAE if not specified
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenTCN, self).__init__(
            cls_model=HiTGenTCN,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config


class AutoHiTGenHier(BaseAuto):

    default_config = {
        "kl_weight": tune.loguniform(1e-5, 1e-1),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [1, 1, 1],
                [2, 2, 2],
                [4, 4, 1],
                [8, 4, 1],
                [16, 8, 1],
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [24, 12, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        # trainer parameters
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,  # Will default to MAE if not specified
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenHier, self).__init__(
            cls_model=HiTGenHier,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)
        return config
