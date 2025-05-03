from logging import getLogger
import tensorflow as tf
from pathlib import Path
import os
import threading
import logging
import keras
import keras_hub
import tensorflow as tf
import json
import numpy as np


logger = getLogger(__name__)
logger.setLevel(logging.INFO)

_model_file_name = "transformer.weights.h5"
_device = "/GPU:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/CPU:0"

content_types = ["application/json", "text/plain"]

logger.info("[INFO] load-thread id: {}".format(threading.current_thread().name))
logger.info("[INFO] load-process id: {}".format(os.getpid()))
logger.info(
    "[INFO] load-SAGEMAKER_MODEL_SERVER_WORKERS: {}".format(
        os.environ.get("SAGEMAKER_MODEL_SERVER_WORKERS", "nonon")
    )
)


logger.info(
    f"[INFO] Container started, model {_model_file_name} will load on device {_device}"
)

_bert_preprocessor: keras_hub.models.Preprocessor = (
    keras_hub.models.BertPreprocessor.from_preset("bert_base_multi")
)
_bert_backbone: keras_hub.models.Backbone = keras_hub.models.BertBackbone.from_preset(
    "bert_base_multi"
)
_tokenizer: keras_hub.models.Tokenizer = _bert_preprocessor.tokenizer

MAX_SEQUENCE_LENGTH = _bert_backbone.get_config()["max_sequence_length"]


def _create_transformer_model():
    # Get the BERT tokenizer and preprocessor
    bert_preprocessor = _bert_preprocessor
    bert_backbone = _bert_backbone
    # Get configuration values
    bertconf = bert_backbone.get_config()
    VOCAB_SIZE = bertconf["vocabulary_size"]
    EMBED_DIM = bertconf["hidden_dim"]

    # Set up the encoder
    bert_preprocessor.trainable = False
    bert_preprocessor.name = "Preprocessor_BERT"
    bert_backbone.trainable = False
    bert_backbone.name = "Encoder_BERT"

    # Create inputs for tokenized data
    token_ids_input = keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="token_ids_input"
    )
    padding_mask_input = keras.layers.Input(
        shape=(None,), dtype=tf.bool, name="padding_mask_input"
    )
    segment_ids_input = keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="segment_ids_input"
    )

    # Create a dictionary of inputs that the BERT backbone expects
    encoder_inputs = {
        "token_ids": token_ids_input,
        "padding_mask": padding_mask_input,
        "segment_ids": segment_ids_input,
    }

    # Pass these inputs directly to the BERT backbone
    encoder_outputs = bert_backbone(encoder_inputs)
    encoder_outputs_seq = encoder_outputs["sequence_output"]

    # Create the encoder model
    encoder = keras.Model(
        [token_ids_input, padding_mask_input, segment_ids_input],
        encoder_outputs_seq,
        name="Bert-multi-encoder",
    )

    # Decoder parameters
    REDUCED_EMBED_DIM = EMBED_DIM // 4
    SHORTENED_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    # Decoder
    decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")

    # Token and position embedding
    shared_embedding = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SHORTENED_SEQUENCE_LENGTH,
        embedding_dim=REDUCED_EMBED_DIM,
    )(decoder_inputs)

    # Dropout layer
    shared_decoder = keras.layers.Dropout(0.7)(shared_embedding)

    # Translation output layer
    decoder_outputs = keras.layers.Dense(
        VOCAB_SIZE, activation="softmax", name="translation_output"
    )(shared_decoder)

    # Create decoder model
    decoder = keras.Model(
        [decoder_inputs, encoder_outputs_seq],
        decoder_outputs,
        name="TranslationDecoder",
    )

    decoder_outputs = decoder([decoder_inputs, encoder_outputs_seq])

    # Language identification
    pooled_output = keras.layers.GlobalAveragePooling1D()(shared_decoder)
    decoder_classification_output = keras.layers.Dense(
        units=2, activation="softmax", name="classification_output"
    )(pooled_output)

    # Create language identification model
    decoder_classification = keras.Model(
        [decoder_inputs, encoder_outputs_seq],
        decoder_classification_output,
        name="LanguageIdentificationDecoder",
    )

    decoder_classification_outputs = decoder_classification(
        [decoder_inputs, encoder_outputs_seq]
    )

    # Create the full transformer model
    transformer = keras.Model(
        [token_ids_input, padding_mask_input, segment_ids_input, decoder_inputs],
        [decoder_outputs, decoder_classification_outputs],
        name="TranslationTransformer",
    )
    return transformer


def model_fn(model_dir, context) -> dict:
    """
    Load the model for inference
    """

    logger.info("[INFO] model_fn-thread id: {}".format(threading.current_thread().name))
    logger.info("[INFO] loading model on device: {}".format(_device))
    logger.info("[INFO] loading model from dir: {}".format(model_dir))

    model_path = Path(model_dir) / _model_file_name
    logger.info(
        "[INFO] model_path: {}, exists: {}".format(model_path, model_path.exists())
    )

    transformer = _create_transformer_model()

    transformer.load_weights(model_path)

    # set inference mode
    transformer.trainable = False

    transformer.compile(jit_compile=False)

    logger.info("[INFO] model loaded successfully")

    return transformer


def input_handler(input_data: bytes, content_type):
    """
    Parse input data payload
    """
    logger.info("[INFO] input_fn-thread id: {}".format(threading.current_thread().name))
    logger.info("[INFO] input_fn-process id: {}".format(os.getpid()))
    logger.info("[INFO] input_fn-content_type: {}".format(content_type))

    if content_type == "application/json":
        obj = json.loads(input_data)
        # a json with a key ""query"
        text = obj["query"]
    elif content_type == "text/plain":
        text = input_data.decode("utf-8")

    else:
        raise ValueError(
            "Content type {} not supported, received {}".format(
                content_type, input_data
            )
        )
    if not isinstance(text, str):
        raise ValueError("Content type {} not supported".format(content_type))
    preprocessed = _bert_preprocessor([text])

    return preprocessed, text


def predict_fn(preprocessed_tuple, model):
    logger.info("predict_fn: Predicting for an image...")

    preprocessed, query = preprocessed_tuple

    token_ids = preprocessed["token_ids"]
    padding_mask = preprocessed["padding_mask"]
    segment_ids = preprocessed["segment_ids"]

    # Initialize with start token
    start_token_id = _tokenizer.token_to_id("[CLS]")
    end_token_id = _tokenizer.token_to_id("[SEP]")

    # Create initial decoder input with start token
    decoder_input = tf.constant([[start_token_id]])

    # Identify the language first
    _, classification_output = model.predict(
        [token_ids, padding_mask, segment_ids, decoder_input], verbose=0
    )

    language_id = tf.argmax(classification_output[0]).numpy()
    language = "English" if language_id == 0 else "Spanish"

    # Generate translation token by token using greedy search
    generated_tokens = [start_token_id]
    token_probabilities = []

    for i in range(MAX_SEQUENCE_LENGTH):
        # Prepare decoder input from generated tokens so far
        decoder_input = tf.constant([generated_tokens])

        # Get model prediction
        translation_output, _ = model.predict(
            [token_ids, padding_mask, segment_ids, decoder_input], verbose=0
        )

        # Get the next token prediction (last position in sequence)
        next_token_logits = translation_output[0, -1, :]
        next_token_probs = tf.nn.softmax(next_token_logits).numpy()

        # Greedy selection - pick the most probable token
        next_token_id = int(np.argmax(next_token_probs))
        next_token_prob = float(next_token_probs[next_token_id])

        # Add the predicted token and its probability
        generated_tokens.append(next_token_id)
        token_probabilities.append(next_token_prob)

        # Stop if we predict the end token
        if next_token_id == end_token_id:
            break

    # Process the generated tokens
    token_strings = []
    for token_id in generated_tokens:
        token = _tokenizer.id_to_token(token_id)
        token_strings.append(token)

    # Create the translation text
    translation = " ".join(token_strings)
    clean_translation = filtered_tokens = [
        t for t in generated_tokens if t not in [start_token_id, end_token_id]
    ]
    clean_translation = _tokenizer.detokenize(filtered_tokens)

    # Calculate overall translation confidence
    avg_confidence = (
        sum(token_probabilities) / len(token_probabilities)
        if token_probabilities
        else 0
    )

    return {
        "language": language,
        "original_text": query,
        "translation": translation,
        "clean_translation": clean_translation,
        "tokens": token_strings,
        "token_ids": generated_tokens,
        "confidence": float(avg_confidence),
    }


def output_handler(prediction: dict, accept) -> bytes:
    logger.info(
        "[INFO] output_fn-thread id: {}".format(threading.current_thread().name)
    )
    logger.info("[INFO] output_fn-process id: {}".format(os.getpid()))

    # TODO
    if accept == "application/json":
        prediction.pop("tokens")
        prediction.pop("token_ids")
        return json.dumps(prediction).encode("utf-8"), accept
    elif accept == "text/plain":
        return prediction["clean_translation"].encode("utf-8"), accept
