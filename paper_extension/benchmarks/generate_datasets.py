# Copyright 2025 Chaitanya Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "src"))

import qml_benchmarks.data as data_module

# --------------------------------------------------------------------
# DEFAULT_PARAMS defines for each dataset:
#   – generator_function_name: name in data_module
#   – base_filename_prefix   : folder/name prefix
#   – configurations         : list of dicts with
#       * iter_params           (varied per run)
#       * static_params         (fixed per run)
#       * n_samples_train/test or n_samples_total
#       * filename_identifier_params: keys to build filename suffix
#       * random_state_offset   : to shift RNG seed
#       * split_shuffle         : for time-series datasets
# --------------------------------------------------------------------

DEFAULT_PARAMS = {
    "bars_and_stripes": {
        "generator_function_name": "generate_bars_and_stripes",
        "base_filename_prefix": "bars_and_stripes",
        "configurations": [
            {
                "iter_params": {"width": size, "height": size},
                "static_params": {"noise_std": 0.5},
                "n_samples_train": 500,
                "n_samples_test": 100,
                "filename_identifier_params": ["width"],
                "random_state_offset": i
            }
            for i, size in enumerate([8, 32])
        ],
    },
    "linearly_separable": {
        "generator_function_name": "generate_linearly_separable",
        "base_filename_prefix": "linearly_separable",
        "configurations": [
            {
                "iter_params": {"n_features": n, "margin": round(0.02 * n, 3)},
                "static_params": {},
                "n_samples_total": 150,
                "filename_identifier_params": ["n_features", "margin"],
                "random_state_offset": i
            }
            for i, n in enumerate([2, 5, 10, 20])
        ],
    },
    "hidden_manifold_model": {
        "generator_function_name": "generate_hidden_manifold_model",
        "base_filename_prefix": "hidden_manifold",
        "configurations": [
            {
                "iter_params": {"n_features": n},
                "static_params": {"manifold_dimension": 6},
                "n_samples_total": 150,
                "filename_identifier_params": ["n_features", "manifold_dimension"],
                "random_state_offset": i
            }
            for i, n in enumerate([2, 5, 10, 20])
        ],
    },
    "hyperplanes_parity": {
        "generator_function_name": "generate_hyperplanes_parity",
        "base_filename_prefix": "hyperplanes_diff",
        "configurations": [
            {
                "iter_params": {"n_hyperplanes": h},
                "static_params": {"n_features": 10, "dim_hyperplanes": 3},
                "n_samples_total": 150,
                "filename_identifier_params": ["n_features", "dim_hyperplanes", "n_hyperplanes"],
                "random_state_offset": i
            }
            for i, h in enumerate([2, 5, 10, 20])
        ],
    },
    "stock_features": {
        "generator_function_name": "generate_stock_features_and_labels",
        "base_filename_prefix": "stock_features",
        "configurations": [
            {
                "iter_params": {},
                "static_params": {"ticker": "AAPL", "start": "2010-01-01", "end": "2024-01-01"},
                "filename_identifier_params": ["ticker"],
                "split_shuffle": False
            }
        ],
    },
    "two_curves": {
        "generator_function_name": "generate_two_curves",
        "base_filename_prefix": "two_curves",
        "configurations": [
            {
                "iter_params": {"n_features": n},
                "static_params": {"degree": 5, "offset": 0.1, "noise": 0.01},
                "n_samples_total": 150,
                "filename_identifier_params": ["n_features", "degree", "offset", "noise"],
                "random_state_offset": i
            }
            for i, n in enumerate([2, 5, 10, 20])
        ],
    },
    "credit_card_fraud": {
        "base_filename_prefix": "credit_card_fraud",
        "configurations": [
            {
                "split_test_size": 0.2,
                "random_state_offset": 0,
                "sample_percentage": 0.02,
                "pca_n_components": 6,
                "select_k_best_features": 2 
            }
        ],
    },
}


class generate_datasets:
    def __init__(self, outdir: Path, params: dict = DEFAULT_PARAMS):
        self.outdir = outdir
        self.params = params
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.base_seed = 42

    def bars_and_stripes(self):
        grp = "bars_and_stripes"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            width = cfg["iter_params"]["width"]
            height = cfg["iter_params"]["height"]
            noise = cfg["static_params"]["noise_std"]
            n_tr = cfg["n_samples_train"]
            n_te = cfg["n_samples_test"]

            X_tr, y_tr = gen_fn(n_tr, height, width, noise)
            X_te, y_te = gen_fn(n_te, height, width, noise)

            suffix = "_".join(f"{k}{cfg['iter_params'][k]}" for k in cfg["filename_identifier_params"])
            ft = subdir / f"bars_and_stripes_{suffix}_train.csv"
            fs = subdir / f"bars_and_stripes_{suffix}_test.csv"

            np.savetxt(ft, np.c_[X_tr.reshape(n_tr, -1), y_tr], delimiter=",")
            np.savetxt(fs, np.c_[X_te.reshape(n_te, -1), y_te], delimiter=",")

    def linearly_separable(self):
        grp = "linearly_separable"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            n_tot = cfg["n_samples_total"]
            n_feat = cfg["iter_params"]["n_features"]
            margin = cfg["iter_params"]["margin"]

            X, y = gen_fn(n_tot, n_feat, margin)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

            suffix = "_".join(f"{k}{cfg['iter_params'][k]}" for k in cfg["filename_identifier_params"])
            ft = subdir / f"linsep_{suffix}_train.csv"
            fs = subdir / f"linsep_{suffix}_test.csv"

            np.savetxt(ft, np.c_[X_tr, y_tr], delimiter=",")
            np.savetxt(fs, np.c_[X_te, y_te], delimiter=",")

    def hidden_manifold_model(self):
        grp = "hidden_manifold_model"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            n_tot = cfg["n_samples_total"]
            n_feat = cfg["iter_params"]["n_features"]
            m_dim = cfg["static_params"]["manifold_dimension"]

            X, y = gen_fn(n_tot, n_feat, m_dim)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

            suffix = "_".join(
                f"{k}{(cfg['iter_params'] | cfg['static_params'])[k]}"
                for k in cfg["filename_identifier_params"]
            )
            ft = subdir / f"hmm_{suffix}_train.csv"
            fs = subdir / f"hmm_{suffix}_test.csv"

            np.savetxt(ft, np.c_[X_tr, y_tr], delimiter=",")
            np.savetxt(fs, np.c_[X_te, y_te], delimiter=",")

    def hyperplanes_parity(self):
        grp = "hyperplanes_parity"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            n_tot = cfg["n_samples_total"]
            params = {**cfg["iter_params"], **cfg["static_params"]}
            X, y = gen_fn(n_tot, **params)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

            suffix = "_".join(f"{k}{params[k]}" for k in cfg["filename_identifier_params"])
            ft = subdir / f"hyperplanes_{suffix}_train.csv"
            fs = subdir / f"hyperplanes_{suffix}_test.csv"

            np.savetxt(ft, np.c_[X_tr, y_tr], delimiter=",")
            np.savetxt(fs, np.c_[X_te, y_te], delimiter=",")

    def stock_features(self):
        grp = "stock_features"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            sp = cfg["static_params"]
            X, y = gen_fn(sp["ticker"], sp["start"], sp["end"])
            shuffle = cfg.get("split_shuffle", True)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, shuffle=shuffle, random_state=seed
            )

            suffix = "_".join(f"{k}{sp[k]}" for k in cfg["filename_identifier_params"])
            ft = subdir / f"stock_{suffix}_train.csv"
            fs = subdir / f"stock_{suffix}_test.csv"

            y_tr_col = y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr
            y_te_col = y_te.reshape(-1, 1) if y_te.ndim == 1 else y_te
            
            data_tr = np.c_[X_tr, y_tr_col]
            data_te = np.c_[X_te, y_te_col]

            pd.DataFrame(data_tr).to_csv(ft, index=False, header=False)
            pd.DataFrame(data_te).to_csv(fs, index=False, header=False)

    def two_curves(self):
        grp = "two_curves"
        cfgs = self.params[grp]["configurations"]
        gen_fn = getattr(data_module, self.params[grp]["generator_function_name"])
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)

            n_tot = cfg["n_samples_total"]
            params = {**cfg["iter_params"], **cfg["static_params"]}
            X, y = gen_fn(n_samples=n_tot, **params)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

            suffix_parts = []
            for k_param in cfg["filename_identifier_params"]:
                value = params[k_param]
                suffix_parts.append(f"{k_param}{str(value).replace('.', 'p')}")
            suffix = "_".join(suffix_parts)
            
            ft = subdir / f"{self.params[grp]['base_filename_prefix']}_{suffix}_train.csv"
            fs = subdir / f"{self.params[grp]['base_filename_prefix']}_{suffix}_test.csv"

            np.savetxt(ft, np.c_[X_tr, y_tr], delimiter=",")
            np.savetxt(fs, np.c_[X_te, y_te], delimiter=",")


    def credit_card_fraud(self):
        grp = "credit_card_fraud"
        cfgs = self.params[grp]["configurations"]
        local_csv_path = repo / "paper_extension" / "datasets" / "creditcard.csv" 

        gen_fn = lambda: data_module.generate_credit_card_fraud_features_and_labels(file_path=local_csv_path)
        subdir = self.outdir / self.params[grp]["base_filename_prefix"]
        subdir.mkdir(exist_ok=True)

        for cfg in cfgs:
            seed = self.base_seed + cfg.get("random_state_offset", 0)
            np.random.seed(seed)
            X, y = gen_fn()
            sample_percentage = cfg.get("sample_percentage", 1.0)

            if sample_percentage < 1.0 and 0 < sample_percentage:
                print(f"Original dataset size: {X.shape[0]} samples.")
                X_sampled, _, y_sampled, _ = train_test_split(
                    X, y, 
                    train_size=sample_percentage, 
                    random_state=seed, 
                    stratify=y
                )
                X, y = X_sampled, y_sampled
                print(f"Subsampled to {sample_percentage*100}%: {X.shape[0]} samples.")

            pca_n_components = cfg.get("pca_n_components")
            if isinstance(pca_n_components, int) and pca_n_components > 0 and pca_n_components < X.shape[1]:
                pca = PCA(n_components=pca_n_components, random_state=seed)
                X = pca.fit_transform(X)
                print(f"Dataset shape after PCA: {X.shape}")

            select_k_best_features = cfg.get("select_k_best_features")
            if isinstance(select_k_best_features, int) and select_k_best_features > 0 and select_k_best_features < X.shape[1]:
                selector = SelectKBest(score_func=f_classif, k=select_k_best_features)
                X = selector.fit_transform(X, y)
                print(f"Dataset shape after SelectKBest: {X.shape}")

            test_size = cfg.get("split_test_size", 0.2)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
            
            ft_name = f"{self.params[grp]['base_filename_prefix']}_train.csv"
            fs_name = f"{self.params[grp]['base_filename_prefix']}_test.csv"
            
            ft = subdir / ft_name
            fs = subdir / fs_name

            pd.DataFrame(np.c_[X_tr, y_tr]).to_csv(ft, index=False, header=False)
            pd.DataFrame(np.c_[X_te, y_te]).to_csv(fs, index=False, header=False)


def main(outdir: Path, params: dict = DEFAULT_PARAMS):
    gen = generate_datasets(outdir, params)
    for ds_name in params:
        fn = getattr(gen, ds_name)
        fn()


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    outdir = root_dir / "datasets_generated"
    main(outdir)