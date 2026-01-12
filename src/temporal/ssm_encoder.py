# src/temporal/ssm_encoder.py
from __future__ import annotations
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, GlobalAveragePooling1D


class DiagonalSSMLayer(Layer):
    """
    Graph-safe diagonal SSM layer.
    Input:  x  (B, T, 1)
    Output: y  (B, T, d_out)
      s_t = sigmoid(A) ⊙ s_{t-1} + x_t @ B
      y_t = s_t @ C
    """
    def __init__(self, d_state: int, d_out: int, name: str = "diag_ssm", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_state = int(d_state)
        self.d_out = int(d_out)

    def build(self, input_shape):
        # input_shape = (B, T, 1)
        self.A = self.add_weight(
            name="A", shape=(self.d_state,),
            initializer="zeros", trainable=True
        )
        self.B = self.add_weight(
            name="B", shape=(1, self.d_state),
            initializer="glorot_uniform", trainable=True
        )
        self.C = self.add_weight(
            name="C", shape=(self.d_state, self.d_out),
            initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, x, training: Optional[bool] = None):
        # x: (B, T, 1)
        x = tf.convert_to_tensor(x)
        a = tf.sigmoid(self.A)                          # (d_state,)
        a = tf.reshape(a, (1, self.d_state))            # (1, d_state), for broadcasting

        # Time-major for tf.scan: (T, B, 1)
        x_tm = tf.transpose(x, [1, 0, 2])
        Bdim = tf.shape(x)[0]

        # scan over timesteps; carry is state s (B, d_state)
        def step(carry, xt):                            # xt: (B, 1)
            s = carry * a + tf.matmul(xt, self.B)       # (B, d_state)
            y = tf.matmul(s, self.C)                    # (B, d_out)
            return s, y

        s0 = tf.zeros((Bdim, self.d_state), dtype=x.dtype)
        # tf.scan returns stacked over time (T, B, *)
        states_tm, outputs_tm = tf.scan(
            fn=step, elems=x_tm, initializer=(s0, tf.zeros((Bdim, self.d_out), dtype=x.dtype))
        )
        # back to batch-major (B, T, d_out)
        y = tf.transpose(outputs_tm, [1, 0, 2])
        return y

    def compute_output_shape(self, input_shape):
        # input_shape: (B, T, 1)
        return (input_shape[0], input_shape[1], self.d_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_state": self.d_state, "d_out": self.d_out})
        return cfg


def _build_ssm_autoencoder(num_bins: int, d_state: int, d_out: int) -> Model:
    """Map (B, T, 1) -> (B, d_out) with SSM + pooling."""
    x_in = Input(shape=(num_bins, 1), name="series")
    h_seq = DiagonalSSMLayer(d_state=d_state, d_out=d_out, name="diag_ssm")(x_in)
    h = GlobalAveragePooling1D(name="pool")(h_seq)  # (B, d_out)
    return Model(inputs=x_in, outputs=h, name="ssm_autoencoder")


def build_ssm_node_embeddings(
    num_bins: int,
    d_state: int,
    d_out: int,
    emb_dim: int,
    dropout: float = 0.1,
    train_df=None,
    node_order=None,
    **kwargs,  # accept and ignore extra kwargs passed by runners
) -> Model:
    """
    Returns compiled model: (B, num_bins, 1) -> (B, emb_dim).
    """
    enc = _build_ssm_autoencoder(num_bins=num_bins, d_state=d_state, d_out=d_out)

    x_in = Input(shape=(num_bins, 1), name="series")
    h = enc(x_in)
    if dropout and dropout > 0:
        h = Dropout(dropout, name="drop")(h)
    z = Dense(emb_dim, activation="linear", name="z")(h)

    model = Model(inputs=x_in, outputs=z, name="ssm_node_embedder")
    model.compile(optimizer="adam", loss="mse")
    return model


def get_default_ssm_hparams():
    return dict(num_bins=128, d_state=64, d_out=64, emb_dim=64, dropout=0.1)
