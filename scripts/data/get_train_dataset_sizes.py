import pprint

from absl import app, flags
from ml_collections import config_flags

from openx.data.core import load_dataset
from openx.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to a config file", lock_config=True)

"""
A simple script for getting the exact size of the train split of datasets.
"""


def main(_):
    dataset_sizes = {}
    for dataset_name, config in FLAGS.config.dataloader.datasets.to_dict().items():
        assert "path" in config and "transform" in config
        transform_fn = ModuleSpec.instantiate(config["transform"])
        filter_fn = ModuleSpec.instantiate(config["filter"]) if config.get("filter", None) is not None else None
        path = config["path"]

        dataset = load_dataset(path, config["train_split"], standardization_transform=transform_fn, filter_fn=filter_fn)

        num_ep, num_steps = dataset.reduce((0, 0), lambda state, ep: (state[0] + 1, state[1] + ep["ep_len"][0] - 1))

        dataset_sizes[dataset_name] = dict(num_ep=num_ep, num_steps=num_steps)

    pprint.pprint(dataset_sizes)


if __name__ == "__main__":
    app.run(main)
