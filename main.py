# cat_dog_classification/
# ├── data/
# │   ├── train/                # 训练集
# │   │   ├── cats/             # 猫的图片（例：cat.0.jpg, cat.1.jpg...）
# │   │   └── dogs/             # 狗的图片（例：dog.0.jpg, dog.1.jpg...）
# │   ├── validation/           # 验证集
# │   │   ├── cats/
# │   │   └── dogs/
# │   └── test/                 # 测试集（可选）
# │       ├── cats/
# │       └── dogs/
# ├── models/                   # 保存训练好的模型
# ├── src/
# │   ├── train.py              # 模型训练脚本
# │   ├── predict.py            # 单张图片预测脚本
# │
# ├── logs/                     # 训练日志和可视化结果
# └── requirements.txt          # 依赖库列表