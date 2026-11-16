"""Xây dựng mạng FaceSeparableCNN và mô hình trích xuất embedding tối ưu cho dữ liệu nhỏ (From Scratch)."""
import keras
from keras import layers
import tensorflow as tf

from .config import IMAGE_SIZE


@keras.saving.register_keras_serializable(package="FaceRecognition")
class L2Normalization(layers.Layer):
    """Chuẩn hóa L2 vector embedding về độ dài đơn vị (unit sphere)."""

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def build_cnn(class_count: int):
    """Xây dựng mạng FaceSeparableCNN tự học từ đầu (From Scratch) với Separable Residual Blocks và L2-normalized embedding."""
    # Tự động tăng cường dữ liệu on-the-fly (chỉ kích hoạt khi train, tự tắt khi inference)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
        layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
        layers.RandomBrightness(0.08),
        layers.RandomContrast(0.08),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="face_input")
    x = augmentation(inputs)

    # Khối Stem: Giảm nhanh chiều không gian xuống 64x64 giúp tăng tốc tính toán trên CPU gấp nhiều lần
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3 tầng Separable Residual Stages
    for filters in [64, 128, 256]:
        res = x
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Skip connection kết hợp 1x1 conv nếu cần match channel
        if res.shape[-1] != filters:
            res = layers.Conv2D(filters, 1, padding="same", use_bias=False)(res)
            res = layers.BatchNormalization()(res)

        x = layers.Add()([res, x])
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

    # Pooling kết hợp Avg + Max để thu thập toàn diện đặc trưng khuôn mặt
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    features = layers.Concatenate()([gap, gmp])
    features = layers.Dropout(0.35)(features)

    # Lớp biểu diễn đặc trưng 128 chiều
    embed_dense = layers.Dense(128, use_bias=False, name="embed_dense")(features)
    embed_bn = layers.BatchNormalization(name="embed_bn")(embed_dense)
    # L2 Normalization chuẩn hóa vector độ dài 1 (tối ưu hóa khoảng cách Cosine trên mặt cầu)
    embedding = L2Normalization(name="embedding")(embed_bn)

    # Đầu ra phân loại danh tính
    drop_out = layers.Dropout(0.25)(embedding)
    outputs = layers.Dense(class_count, activation="softmax", name="classification")(drop_out)

    classifier = keras.Model(inputs, outputs, name="face_cnn")
    extractor = keras.Model(inputs, embedding, name="cnn_feature_extractor")

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return classifier, extractor


