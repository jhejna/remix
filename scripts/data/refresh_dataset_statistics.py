from absl import app, flags
from ml_collections import config_flags

from openx.data.core import compute_dataset_statistics
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to a config file", lock_config=True)

"""
A simple script for refreshing the dataset statistics of a given config.
"""


def main(_):
    for _, config in FLAGS.config.dataloader.datasets.to_dict().items():
        assert "path" in config and "transform" in config
        transform_fn = ModuleSpec.instantiate(config["transform"])
        compute_dataset_statistics(config["path"], transform_fn, recompute_statistics=True, save_statistics=True)


if __name__ == "__main__":
    app.run(main)
