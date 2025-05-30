import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体（根据步骤1结果修改）
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def predict_image(model_path):
    # 创建Tkinter根窗口并隐藏
    root = tk.Tk()
    root.withdraw()

    # 打开文件选择对话框
    print("请选择要预测的图片...")
    img_path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    # 如果用户取消选择，退出函数
    if not img_path:
        print("已取消选择")
        return

    # 加载模型
    model = tf.keras.models.load_model(model_path)

    # 加载并预处理图片
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 预测
    prediction = model.predict(img_array)
    class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
    confidence = prediction[0][0] if class_label == 'Dog' else 1 - prediction[0][0]

    # 显示结果
    plt.imshow(img)
    plt.title(f'预测结果: {class_label} (置信度: {confidence * 100:.2f}%)')
    plt.axis('off')
    plt.show()

    # 返回预测结果
    return {
        'image_path': img_path,
        'predicted_class': class_label,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    model_path = '../models/cat_dog_model.h5'
    result = predict_image(model_path)
    if result:
        print(f"预测结果: {result['predicted_class']} (置信度: {result['confidence'] * 100:.2f}%)")
        print(f"图片路径: {result['image_path']}")