import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['aes', 'raft', 'videoclip', 'videophy']
times = [0.9487, 3.7776, 0.9015, 9.6574]

# 科研风格配色（类似 ColorBrewer / Seaborn）
scientific_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# 位置 & 宽度
x = np.arange(len(models))
bar_width = 0.6  # 控制柱子变细

# 绘图
plt.figure(figsize=(8, 5))
bars = plt.bar(x, times, width=bar_width, color=scientific_colors)

# 在柱状图上标注时间
for bar, time in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             f"{time:.4f}",
             ha='center', va='bottom', fontsize=10)

# 坐标轴 & 样式
plt.xticks(x, models)
plt.ylabel("Execution Time (s)")
plt.title("Execution Time of Reward Models")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图片
plt.savefig("/nvfile-heatstorage/tele_data_share/wyb/Dancegrpo/recipe/dancegrpo/plot/reward_model_times.png")

