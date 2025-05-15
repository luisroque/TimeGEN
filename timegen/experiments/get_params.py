import os
import pandas as pd
from neuralforecast.auto import (
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
)
from timegen.model_pipeline.auto.AutoModels import (
    AutoTimeGEN,
    AutoTimeGEN_S,
    AutoTimeGEN_M,
    AutoTimeGEN_D,
)
from timegen.model_pipeline.core.core_extension import CustomNeuralForecast

BASE_DIR = "assets/model_weights_out_domain/hypertuning"
OUTPUT_CSV = "model_parameter_counts.csv"
results_out_dir = "assets/results_forecast_out_domain_summary"


MODEL_CLASSES = [
    ("AutoTimeGEN", AutoTimeGEN),
    ("AutoTimeGEN_S", AutoTimeGEN_S),
    ("AutoTimeGEN_M", AutoTimeGEN_M),
    ("AutoTimeGEN_D", AutoTimeGEN_D),
    ("AutoNHITS", AutoNHITS),
    ("AutoKAN", AutoKAN),
    ("AutoPatchTST", AutoPatchTST),
    ("AutoiTransformer", AutoiTransformer),
    ("AutoTSMixer", AutoTSMixer),
    ("AutoTFT", AutoTFT),
]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    records = []

    for filename in os.listdir(BASE_DIR):
        if filename.endswith("_neuralforecast") and filename.startswith("MIXED"):
            for name, ModelClass in MODEL_CLASSES:
                if name not in filename:
                    continue

                model_path = os.path.join(BASE_DIR, filename)

                try:
                    # dummy init for loading
                    try:
                        auto_model = ModelClass(h=1, num_samples=1, config={})
                    except:
                        auto_model = ModelClass(
                            h=1, num_samples=1, config={}, n_series=1
                        )

                    nf = CustomNeuralForecast(models=[auto_model], freq="D")
                    nf = nf.load(path=model_path)

                    torch_model = nf.models[0]
                    n_params = count_parameters(torch_model)

                    records.append(
                        {
                            "model_name": name,
                            "path": model_path,
                            "num_parameters": n_params,
                        }
                    )

                    print(f"{name} — {n_params:,} parameters")

                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")

    df = pd.DataFrame(records)
    df_agg = df.groupby("model_name", as_index=False)["num_parameters"].mean()

    csv_path = os.path.join(results_out_dir, OUTPUT_CSV)
    df_agg.to_csv(csv_path, index=False)

    print(f"\nSaved model parameter counts to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
