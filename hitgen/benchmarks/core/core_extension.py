import fsspec
import pickle

from hitgen.benchmarks.HiTGen import HiTGen
from hitgen.benchmarks.HiTGenDeep import HiTGenDeep


from neuralforecast import NeuralForecast
from neuralforecast import core as nf_core

nf_core.MODEL_FILENAME_DICT.update(
    {
        "autohitgen": HiTGen,
        "autohitgendeep": HiTGenDeep,
    }
)


class CustomNeuralForecast(NeuralForecast):
    """
    NeuralForecast that overrides the static load() method
    to allow loading custom (user-defined) models
    """

    @staticmethod
    def load(
        path: str,
        verbose: bool = False,
        **kwargs,
    ) -> "CustomNeuralForecast":
        """
        Load a NeuralForecast object from disk, allowing for user-defined models.
        """
        if path.endswith("/"):
            path = path[:-1]

        fs, _, _ = fsspec.get_fs_token_paths(path)
        files = [f.split("/")[-1] for f in fs.ls(path) if fs.isfile(f)]

        models_ckpt = [f for f in files if f.endswith(".ckpt")]
        if len(models_ckpt) == 0:
            raise FileNotFoundError(f"No .ckpt model files found in directory: {path}")

        if verbose:
            print(10 * "-" + " Loading models " + 10 * "-")

        try:
            with fsspec.open(f"{path}/alias_to_model.pkl", "rb") as f:
                alias_to_model = pickle.load(f)
        except FileNotFoundError:
            alias_to_model = {}

        models = []
        for ckpt_file in models_ckpt:
            model_name = "_".join(ckpt_file.split("_")[:-1])
            model_class_name = alias_to_model.get(model_name, model_name)

            if model_class_name.lower() in nf_core.MODEL_FILENAME_DICT:
                cls_ = nf_core.MODEL_FILENAME_DICT[model_class_name.lower()]
            else:
                raise ValueError(
                    f"Could not find model_class_name='{model_class_name}' in Nixtla's "
                    "core models or in custom_model_dict. "
                    f"Ensure you have spelled '{model_class_name}' correctly or "
                    "provided it in custom_model_dict."
                )

            # load the actual checkpoint file
            loaded_model = cls_.load(f"{path}/{ckpt_file}", **kwargs)
            loaded_model.alias = model_name
            models.append(loaded_model)

            if verbose:
                print(f"Model '{model_name}' loaded as '{model_class_name}'.")

        dataset = None
        if verbose:
            print(10 * "-" + " Loading dataset " + 10 * "-")
        try:
            with fsspec.open(f"{path}/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            if verbose:
                print("Dataset loaded.")
        except FileNotFoundError:
            if verbose:
                print("No dataset found in directory. (Skipping)")

        if verbose:
            print(10 * "-" + " Loading configuration " + 10 * "-")
        try:
            with fsspec.open(f"{path}/configuration.pkl", "rb") as f:
                config_dict = pickle.load(f)
            if verbose:
                print("Configuration loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(
                "No configuration.pkl file found in directory. Cannot restore metadata."
            )

        freq = config_dict.get("freq", "D")
        local_scaler_type = config_dict.get("local_scaler_type", None)
        my_nf = CustomNeuralForecast(
            models=models,
            freq=freq,
            local_scaler_type=local_scaler_type,
        )

        my_nf.id_col = config_dict.get("id_col", "unique_id")
        my_nf.time_col = config_dict.get("time_col", "ds")
        my_nf.target_col = config_dict.get("target_col", "y")

        if dataset is not None:
            my_nf.dataset = dataset
            for attr in ["uids", "last_dates", "ds"]:
                setattr(my_nf, attr, config_dict.get(attr, None))

        my_nf._fitted = config_dict.get("_fitted", False)
        my_nf.scalers_ = config_dict.get("scalers_", {})

        my_nf.prediction_intervals = config_dict.get("prediction_intervals", None)
        my_nf._cs_df = config_dict.get("_cs_df", None)

        return my_nf
