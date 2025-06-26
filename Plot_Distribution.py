import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = 'Steam_Twitch_Metacritic_2024.xlsx'
df = pd.read_excel(file_path, sheet_name='Steam Games Raw')

# 选取所需列
peak_ccu = df['PeakCCU'].dropna()  # 确保不包含缺失值
total_reviews = df['TotalReviews'].dropna()

# 创建分布图
plt.figure(figsize=(14, 6))

# Peak CCU 的分布
plt.subplot(1, 2, 1)
sns.histplot(peak_ccu, bins=30, kde=True, color='blue', alpha=0.6)
plt.title('Distribution of Peak CCU')
plt.xlabel('Peak CCU')
plt.ylabel('Frequency')

# Total Reviews 的分布
plt.subplot(1, 2, 2)
sns.histplot(total_reviews, bins=30, kde=True, color='green', alpha=0.6)
plt.title('Distribution of Total Reviews')
plt.xlabel('Total Reviews')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()