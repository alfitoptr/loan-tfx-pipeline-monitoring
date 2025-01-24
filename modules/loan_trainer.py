import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft
import os
import logging
import keras_tuner as kt
from tfx.components.tuner.component import TunerFnResult

num_features = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
cat_features = {
    'person_home_ownership': 3,
    'previous_loan_defaults_on_file': 2
}
target = 'loan_status'

def build_model(hp):
    """

    """
    input_features = []
    for key, dim in cat_features.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=f"{key}_ohe")
        )
    
    for feature in num_features:
        input_features.append(
            tf.keras.Input(shape=(1,), name=feature)
        )

    concatenate = layers.Concatenate()(input_features)

    x = layers.Dense(hp.Int('unit1', min_value=32, max_value=128, step=32), activation='relu')(concatenate)
    x = layers.Dense(hp.Int('unit2', min_value=32, max_value=128, step=32), activation='relu')(x)
    x = layers.Dense(hp.Int('unit3', min_value=32, max_value=128, step=32), activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input_features, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
 
    model.tft_layer = tf_transform_output.transform_features_layer()
 
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(target)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
 
        transformed_features = model.tft_layer(parsed_features)
 
        outputs = model(transformed_features)
        return {"outputs": outputs}
 
    return serve_tf_examples_fn
 
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of 
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
 
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=target,
    )
 
    return dataset

def tuner_fn(fn_args):
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args used to tune models as name/value pairs.
 
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # Memuat training dan validation dataset yang telah di-preprocessing
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  train_set = input_fn(fn_args.train_files[0], tf_transform_output)
  val_set = input_fn(fn_args.eval_files[0], tf_transform_output)
  
  stop_early=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
  # Mendefinisikan strategi hyperparameter tuning
  tuner = kt.RandomSearch(build_model,
                     objective='val_binary_accuracy',
                     max_trials=3,
                     executions_per_trial=1,
                     directory='output/tuning_resuls')
 
  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
  )

def run_fn(fn_args):
    """
    Define the TFX training pipeline.
    Args:
        fn_args: Arguments passed by TFX.
    """
    logging.info("Starting the training pipeline...")
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
 
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)
    
    tuner_result = tuner_fn(fn_args)
    tuner = tuner_result.tuner
    tuner.search(train_dataset, validation_data=eval_dataset, epochs=10)
    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.summary()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
 

    logging.info("Training the model...")

    best_model.fit(
        train_dataset,
        steps_per_epoch=5000,
        validation_data=eval_dataset,
        validation_steps=1000,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            best_model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    best_model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
        )
    
    plot_model(
        best_model, 
        to_file='images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )