import tensorflow as tf
import tensorflow_transform as tft

num_features = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
cat_features = {
    'person_home_ownership': 3,  # Number of unique categories
    'previous_loan_defaults_on_file': 2
}
target = 'loan_status'

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        label_tensor: Tensor of integers (e.g., 0 or 1)
        num_labels: Number of classes
    Returns:
        One-hot encoded tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    outputs = {}

    # One-hot encode categorical features
    for key in cat_features:
        dim = cat_features[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[f'{key}_ohe'] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    # Scale numeric features
    for feature in num_features:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    outputs[target] = tf.cast(inputs[target], tf.int64)
    return outputs