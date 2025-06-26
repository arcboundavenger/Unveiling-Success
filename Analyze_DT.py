import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam_Twitch_Metacritic_games.xlsx', sheet_name='Steam Games')

# 选择目标变量和特征
y = data['LnRevenue']
X = data.drop(columns=['LnRevenue', 'AppID', 'Estimated owners', 'Release date'])

# 确保所有数据都是数值类型
X = X.apply(pd.to_numeric, errors='coerce')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 创建并训练模型（使用默认参数）
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型使用 R^2
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# 计算 NRMSE
nrmse = rmse / (y_test.max() - y_test.min())
print(f'NRMSE: {nrmse}')

# 计算 MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')

# 计算残差
residuals = y_test - y_pred

# 创建一个包含两个子图的图形，横向排列
fig, axs = plt.subplots(1, 2, figsize=(12, 12))  # 确保图形的高度和宽度相同

# 绘制残差图
axs[0].scatter(y_pred, residuals, alpha=0.5)
axs[0].axhline(0, color='red', linestyle='--')
axs[0].set_title('Residual Plot')
axs[0].set_xlabel('Predicted Values')
axs[0].set_ylabel('Residuals')
axs[0].grid()

# 绘制预测误差图
axs[1].scatter(y_test, y_pred, alpha=0.5)
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
axs[1].set_title('Prediction Error Plot')
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Predicted Values')
axs[1].grid()

# 调整每个子图的大小为正方形
for ax in axs:
    ax.set_aspect('equal', adjustable='datalim')

# 显示图形
plt.tight_layout()
plt.show()

# 计算特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # 降序索引

# 创建特征重要性图，横向条形图
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.barh(range(X.shape[1]), importances[indices], align='center')
plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
plt.xlim([0, importances.max() + 0.05])  # 添加一点额外空间在右侧
plt.xlabel('Importance')
plt.ylabel('Features')

# 颠倒y轴顺序
plt.gca().invert_yaxis()  # 这里颠倒y轴，使得最重要的特征在顶部

plt.show()