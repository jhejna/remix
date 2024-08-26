import tensorflow as tf


def filter_by_ep_path(ep, search_strings, path_key: str = "file_path"):
    string = ep["episode_metadata"][path_key]
    if isinstance(search_strings, list):
        bool_tensor = tf.concat(
            [tf.strings.regex_full_match(string, pattern=".*" + s + ".*") for s in search_strings], axis=0
        )
        return tf.math.reduce_all(bool_tensor)
    else:
        return tf.strings.regex_full_match(string, pattern=".*" + search_strings + ".*")
