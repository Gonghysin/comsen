import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 检查模型文件是否存在
model_path = "saved_models/house_price_rf"
if not os.path.exists(model_path):
    print(f"错误：模型文件夹 {model_path} 不存在！")
    print("请先运行训练模型的代码 tran_model.py")
    exit(1)

# 加载保存的模型
try:
    loaded_model = tf.saved_model.load(model_path)
except Exception as e:
    print(f"加载模型失败: {e}")
    exit(1)

# 检查测试数据文件是否存在
test_file_path = "house-prices-advanced-regression-techniques/test.csv"
if not os.path.exists(test_file_path):
    print(f"错误：测试数据文件 {test_file_path} 不存在！")
    exit(1)

# 加载测试数据
test_df = pd.read_csv(test_file_path)

# 处理数据类型，确保与训练数据一致
float_cols = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in float_cols:
    test_df[col] = test_df[col].astype('float32')

# 处理缺失值
numeric_columns = test_df.select_dtypes(include=['float32', 'float64', 'int64']).columns
test_df[numeric_columns] = test_df[numeric_columns].fillna(0)
categorical_columns = test_df.select_dtypes(include=['object']).columns
test_df[categorical_columns] = test_df[categorical_columns].fillna('Missing')

# 将测试数据转换为TensorFlow数据集格式
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_df,
    task=tfdf.keras.Task.REGRESSION
)

# 进行预测
try:
    # 准备输入数据
    input_data = {}
    for feature_name, feature_tensor in test_ds.element_spec[0].items():
        # 获取第一个批次的数据
        for batch in test_ds.take(1):
            input_data[feature_name] = batch[0][feature_name]
    
    # 使用模型进行预测
    predictions = loaded_model.signatures["serving_default"](**input_data)
    predictions = predictions['output_0'].numpy()
    
except Exception as e:
    print(f"预测失败: {e}")
    exit(1)

# 将预测结果添加到数据框中
try:
    results_df = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions.flatten()  # 将预测结果展平
    })

    # 确保预测值为正数
    results_df['SalePrice'] = results_df['SalePrice'].clip(lower=0)

    # 保存预测结果到CSV文件
    results_df.to_csv('predictions.csv', index=False)
    print("预测完成！结果已保存到 predictions.csv")

    # 显示前几个预测结果
    print("\n前5个预测结果：")
    print(results_df.head())

    # 显示预测结果的基本统计信息
    print("\n预测结果统计信息：")
    print(results_df['SalePrice'].describe())

    # 绘制预测结果的分布图
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['SalePrice'], bins=50)
    plt.title('预测房价分布')
    plt.xlabel('预测房价')
    plt.ylabel('频率')
    plt.show()

except Exception as e:
    print(f"处理预测结果时出错: {e}")
    exit(1)
