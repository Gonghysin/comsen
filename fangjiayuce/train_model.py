import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 创建 images 文件夹（如果不存在）
if not os.path.exists("images"):
    os.makedirs("images")

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# 加载训练数据集
train_file_path = "house-prices-advanced-regression-techniques/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("训练数据集形状：{}".format(dataset_df.shape))

# 显示数据集的基本信息
print("\n数据集基本信息：")
print(dataset_df.info())

# 显示数据集的前几行
print("\n数据集预览：")
print(dataset_df.head(3))

# 显示目标变量'SalePrice'的基本统计信息
print("\n房价统计信息：")
print(dataset_df['SalePrice'].describe())

# 绘制房价分布图
# print(dataset_df['SalePrice'].describe())
# plt.figure(figsize=(9, 8))
# sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})
# plt.show()

# 显示数据集中所有不同的数据类型
print("\n数据集中的数据类型：")
print(list(set(dataset_df.dtypes.tolist())))

# 选择数值型特征
df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
print("\n数值型特征预览：")
print(df_num.head())

# 绘制数值型特征的直方图
# plt.figure(figsize=(10, 8))
# df_num.hist(figsize=(10, 8), bins=30, xlabelsize=6, ylabelsize=6)
# plt.tight_layout()  # 调整子图之间的间距
# plt.show()

# 导入numpy
import numpy as np

# 处理数据类型
float_cols = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in float_cols:
    dataset_df[col] = dataset_df[col].astype('float32')

# 处理缺失值
numeric_columns = dataset_df.select_dtypes(include=['float32', 'float64', 'int64']).columns
dataset_df[numeric_columns] = dataset_df[numeric_columns].fillna(0)
categorical_columns = dataset_df.select_dtypes(include=['object']).columns
dataset_df[categorical_columns] = dataset_df[categorical_columns].fillna('Missing')

# 将数据集分为训练集和验证集
def split_dataset(dataset, valid_ratio=0.30):
    valid_indices = np.random.rand(len(dataset)) < valid_ratio
    return dataset[~valid_indices], dataset[valid_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in validation.".format(
    len(train_ds_pd), len(valid_ds_pd)))


# 将数据集从Pandas格式转换为TensorFlow数据集格式
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)

# 显示所有可用的模型
print("\n可用的模型列表：")
print(tfdf.keras.get_all_models())

# 创建随机森林模型
# 创建随机森林模型对象
# hyperparameter_template="benchmark_rank1" 使用预定义的高性能超参数模板
# task=tfdf.keras.Task.REGRESSION 指定这是一个回归任务
rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)

# 编译模型
# metrics=["mse"] 设置均方误差(Mean Squared Error)作为评估指标
# mse用于衡量预测值与实际值之间的平均平方差
rf.compile(metrics=["mse"]) # 可选配置,用于包含评估指标列表

# 训练随机森林模型
print("\n开始训练随机森林模型...")
rf.fit(x=train_ds)

# 可视化随机森林中的第一棵决策树
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# 获取并打印OOB分数
print("\nOOB评分:")
print(rf.summary())

# 获取详细的OOB指标
evaluation = rf.make_inspector().evaluation()
print("\n详细的评估指标:")
print(f"RMSE (均方根误差): {evaluation.rmse}")


# 在验证集上评估模型
print("\n在验证集上的评估结果:")
evaluation = rf.evaluate(x=valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# 显示变量重要性
print("\n特征重要性分析 (NUM_AS_ROOT):")
importances = rf.make_inspector().variable_importances()["NUM_AS_ROOT"]
for feature_name, importance in importances:
    print(f"{feature_name}: {importance:.4f}")


# 可视化前10个最重要的特征
plt.figure(figsize=(10, 6))
features = [str(imp[0]) for imp in importances[:10]]
scores = [float(imp[1]) for imp in importances[:10]]

plt.barh(range(len(features)), scores)
plt.yticks(range(len(features)), features)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top 10 Most Important Features (NUM_AS_ROOT)')
plt.tight_layout()
plt.savefig('images/top_10_features.png')  # 保存图片
plt.close()  # 关闭图形

# 另一种特征重要性可视化
plt.figure(figsize=(12, 8))
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = rf.make_inspector().variable_importances()[variable_importance_metric]

feature_names = [str(vi[0]) for vi in variable_importances]
feature_importances = [float(vi[1]) for vi in variable_importances]
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel('Importance Score')
plt.title('Feature Importance Analysis')
plt.tight_layout()
plt.savefig('images/feature_importance_full.png')  # 保存图片
plt.close()  # 关闭图形

# 训练过程中的RMSE变化曲线
logs = rf.make_inspector().training_logs()
plt.figure(figsize=(10, 6))
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of Trees")
plt.ylabel("RMSE (out-of-bag)")
plt.title("RMSE Evolution During Training")
plt.savefig('images/rmse_evolution.png')  # 保存图片
plt.close()  # 关闭图形

# 在保存模型之前，添加预测代码
print("\n开始预测测试集...")

# 加载测试数据
test_file_path = "house-prices-advanced-regression-techniques/test.csv"
test_df = pd.read_csv(test_file_path)

# 使用与训练数据相同的预处理步骤
# 处理数据类型
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

# 使用训练好的模型进行预测
predictions = rf.predict(test_ds)

# 将预测结果添加到数据框中
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
plt.title('Predicted House Price Distribution')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.savefig('images/price_distribution.png')  # 保存图片
plt.close()  # 关闭图形

# 添加新的可视化：预测结果的箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(results_df['SalePrice'])
plt.title('Predicted House Price Box Plot')
plt.ylabel('Price ($)')
plt.savefig('images/price_boxplot.png')  # 保存图片
plt.close()  # 关闭图形

# 保存详细的预测结果
detailed_results = pd.concat([
    test_df[['Id', 'MSSubClass', 'OverallQual', 'GrLivArea', 'YearBuilt']],
    results_df['SalePrice']
], axis=1)
detailed_results.to_csv('predictions_detailed.csv', index=False)

print("\n所有图表已保存到 images 文件夹")
print("详细预测结果已保存到 predictions_detailed.csv")



# # 最后再保存模型（如果需要的话）
# if not os.path.exists("saved_models"):
#     os.makedirs("saved_models")

# model_save_path = "saved_models/house_price_rf"
# try:
#     rf.save(model_save_path)
# except Exception as e:
#     print(f"保存模型失败: {e}")
    
# print(f"\n模型已保存到: {model_save_path}")

