# 导入必要的库
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('Steam_Twitch_Metacritic_games.xlsx', sheet_name='Steam Games')

# 处理缺失值和无穷大
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# 选择目标变量和特征
y = data['LnRevenue']
X = data.drop(columns=['LnRevenue', 'AppID', 'Estimated owners', 'Release date'])
X = sm.add_constant(X)  # 添加常数项

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 拟合模型
model = sm.OLS(y_train, X_train).fit()

# 打印模型摘要
print(model.summary())

# 计算预测值和评估指标
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
nrmse = rmse / (y_test.max() - y_test.min())
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}, NRMSE: {nrmse}, MAE: {mae}")

# 输出 R² 和 F 值
r_squared = model.rsquared
f_value = model.fvalue
print(f"R²: {r_squared}, F: {f_value}")

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

# 处理 p 值和显著性以及其他统计信息
conf_int = model.conf_int()  # 获取置信区间
results_df = pd.DataFrame({
    'Parameter': model.params.index,
    'Estimate (coef)': model.params.values,
    'Std Err': model.bse.values,
    't': model.tvalues.values,
    'P-value': model.pvalues.values,
    'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in model.pvalues],
    'LLCI': conf_int[0].values,  # 下限
    'ULCI': conf_int[1].values,  # 上限
    'R²': r_squared,              # R²
    'F': f_value                  # F 值
})

# 保存结果
results_df.to_excel('model_results.xlsx', index=False)
print("模型结果已保存为 'model_results.xlsx'.")