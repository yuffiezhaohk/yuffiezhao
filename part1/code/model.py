import tensorflow as tf
from tensorflow.keras import layers, models

def build_deep_halo(
    universe_size: int,
    hidden_width: int,
    network_depth: int
) -> models.Model:
    """
    Constructs the DeepHalo neural network architecture designed for featureless discrete choice.

    Key Architectural Features:
    1.  **Quadratic Activation**: Uses x^2 instead of ReLU to explicitly model interaction effects.
    2.  **ResNet Topology**: Uses skip connections for gradient flow.
    3.  **Availability Masking**: Ensures probability of unavailable items is strictly zero.

    Args:
        universe_size (int): Size of the item universe (Input/Output dimension).
        hidden_width (int): Number of neurons in hidden layers.
        network_depth (int): Total number of non-linear transformations.

    Returns:
        models.Model: A compiled Keras functional model.
    """

    # Input: Binary vector indicating item availability
    inputs = layers.Input(shape=(universe_size,), name='Availability_Input')

    # Initial Projection: Linear mapping to hidden dimension
    # Note: use_bias=False is used to adhere strictly to parameter budget constraints logic
    x = layers.Dense(hidden_width, use_bias=False, name='Projection_Layer')(inputs)

    # Stacked Residual Blocks
    for i in range(1, network_depth + 1):
        # Identity path
        shortcut = x

        # Quadratic Activation: Explicitly models pairwise interactions
        x = layers.Lambda(lambda t: tf.square(t), name=f'Quadratic_Activation_{i}')(x)

        # Linear Transformation (Mixing)
        x = layers.Dense(hidden_width, use_bias=False, name=f'Mixing_Layer_{i}')(x)

        # Residual Connection
        x = layers.Add(name=f'Residual_Add_{i}')([shortcut, x])

    # Output Projection: Map back to item universe size
    logits = layers.Dense(universe_size, use_bias=False, name='Logit_Projection')(x)

    # --- Masking Mechanism ---
    # We must ensure the model cannot assign probability to items not in the input set.
    # 1. Cast input (0/1 floats) to boolean mask
    mask_boolean = layers.Lambda(
        lambda t: tf.cast(t, dtype=tf.bool),
        name='Create_Boolean_Mask'
    )(inputs)

    # 2. Apply Mask: Replace logits of unavailable items with -1e9 (approx negative infinity)
    # This ensures exp(logit) is effectively 0 during Softmax
    masked_logits = layers.Lambda(
        lambda args: tf.where(args[0], args[1], -1e9),
        output_shape=(universe_size,),
        name='Apply_Availability_Mask'
    )([mask_boolean, logits])

    # Probability Output
    outputs = layers.Softmax()(masked_logits)

    return models.Model(inputs=inputs, outputs=outputs)