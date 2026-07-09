"""INR model definition: TCNN hash grid encoding + fully-fused MLP."""

import tinycudann as tcnn


class INR_TCNN:
    """TCNN-accelerated INR: multi-resolution hash grid + fully-fused MLP."""

    def __new__(cls, n_input_dims=3, n_output_dims=1, seed=1337,
                n_hidden_layers=4, n_neurons=16,
                n_levels=16, n_features_per_level=8,
                log2_hashmap_size=19, base_resolution=4,
                per_level_scale=1.5,
                activation="ReLU", output_activation="None"):
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
            "feedback_alignment": False,
        }
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=encoding_config,
            network_config=network_config,
            seed=seed,
        )
        model.encoding_config = encoding_config
        model.network_config = network_config
        return model
