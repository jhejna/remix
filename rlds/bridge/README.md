# Bridge Dataset

A dataset builder for datasets collected with the [bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot).

This code was taken and then modified from Kevin Black's [dlimp](https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py) repository.

To build a particular dataset, cd into its corresponding directory and run `CUDA_VISIBLE_DEVICES="" tfds build --manual_dir <path_to_raw_data> --data_dir <path to save>`. See individual dataset documentation for how to obtain the raw data. You may also want to modify settings inside the <dataset_name>_dataset_builder.py file (e.g., NUM_WORKERS and CHUNKSIZE.)
