import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Baseline (FP16)', 'Quantized (Q8_0)', 'Aggressive (Q4_0)']
memory = [224.0, 119.0, 63.0]  # MiB
speed = [379.51, 364.89, 366.35]  # Tokens/s

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制显存 (左轴 - 柱状图)
color = 'tab:blue'
ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax1.set_ylabel('KV Cache Memory (MiB)', color=color, fontsize=12, fontweight='bold')
bars = ax1.bar(x, memory, width, color=color, alpha=0.7, label='Memory Usage')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 250)

# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height} MiB\n(-{int((224-height)/224*100)}%)' if height!=224 else f'{height} MiB',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 绘制速度 (右轴 - 折线图)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Prompt Processing Speed (T/s)', color=color, fontsize=12, fontweight='bold')
line = ax2.plot(x, speed, color=color, marker='o', linewidth=2, label='Speed')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(300, 450)

# 设置X轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)

plt.title('Impact of KV Cache Quantization on Memory & Speed', fontsize=14, pad=20)
fig.tight_layout()
plt.savefig('kv_cache_benchmark.png', dpi=300)
print("图表已生成: kv_cache_benchmark.png")