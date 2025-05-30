import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# 数据路径
train_dir = '../data/train'
validation_dir = '../data/validation'

# 图像预处理参数
img_width, img_height = 150, 150
batch_size = 32
epochs = 50

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 归一化
    rotation_range=40,  # 随机旋转角度
    width_shift_range=0.2,  # 水平平移
    height_shift_range=0.2,  # 垂直平移
    shear_range=0.2,  # 剪切变换
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest'  # 填充策略
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# 生成训练和验证数据流
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # 二分类
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# 构建CNN模型 - 卷积神经网络训练
model = Sequential([
    # 卷积层 + 非线性层
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    # 池化层
    MaxPooling2D(2, 2),

    # 卷积层 + 非线性层
    Conv2D(64, (3, 3), activation='relu'),
    # 池化层
    MaxPooling2D(2, 2),

    # 卷积层 + 非线性层
    Conv2D(128, (3, 3), activation='relu'),
    # 池化层
    MaxPooling2D(2, 2),

    # 将多维特征展平为一维向量
    Flatten(),

    # 全连接层 + 非线性层
    Dense(512, activation='relu'),

    # 输出层 - 二分类问题
    Dense(1, activation='sigmoid')  # 输出层（0: 猫，1: 狗）
])

# 编译模型
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# 模型保存回调
checkpoint = ModelCheckpoint(
    '../models/cat_dog_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint]
)


# 绘制训练曲线
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.savefig('../logs/training_metrics.png')
    plt.show()


plot_training_history(history)