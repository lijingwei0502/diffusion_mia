import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300

plt.figure(figsize=(10, 8))
dir = 'results.csv'
df = pd.read_csv(dir, header=None, sep=',')  # 确保使用正确的文件路径和分隔符

# 选择固定列和变量列的索引
fixed_columns = [3, 4]  
variable_column = 2

# 固定目标值
fixed_values = [10, 'nn']  # 这些值需要根据您的具体数据进行调整

# 筛选数据
# 这里使用布尔索引来筛选符合固定列值的行
mask = (df.iloc[:, fixed_columns[0]] == fixed_values[0]) & \
       (df.iloc[:, fixed_columns[1]] == fixed_values[1])
filtered_df = df[mask]

print(filtered_df)
# 在进行排序之前，确保变量列是浮点数类型
filtered_df.loc[:, variable_column] = filtered_df.iloc[:, variable_column].astype(float)

# 现在，filtered_df已经根据变量列的浮点值排序
filtered_df = filtered_df.sort_values(by=filtered_df.columns[variable_column])

# 根据某一列的值进行筛选
def filter_and_convert(df, column, value):
    filtered = df[df.iloc[:, column] == value]
    for i in change_column:
        filtered.loc[:, i] = filtered.iloc[:, i].astype(float)
    return filtered

change_column = [6, 7, 8]
naive_df = filter_and_convert(filtered_df, 5, 1)
pian_df = filter_and_convert(filtered_df, 5, 5)
pia_df = filter_and_convert(filtered_df, 5, 10)
secmi_df = filter_and_convert(filtered_df, 5, 20)

average_numbers = naive_df.iloc[:, variable_column]

# 创建一个新的等距横坐标映射
mapping = {20: 1, 50: 2, 100: 3}

# 检查average_numbers中的值是否都在mapping中
if not all(lr in mapping for lr in average_numbers):
    raise ValueError("average_numbers中包含未在mapping中定义的值")

# 将average_numbers的值映射到新的等距值
mapped_average_number = np.array([mapping[lr] for lr in average_numbers])

# 打印数据的长度以进行调试
print("mapped_average_number length:", len(mapped_average_number))
print("naive_df length:", len(naive_df))
print("pian_df length:", len(pian_df))
print("pia_df length:", len(pia_df))
print("secmi_df length:", len(secmi_df))

def plot_data(mapped_average_number, dfs, col, ylabel, filename):
    colors = ['blue', 'orange', 'green', 'red']
    labels = ['1', '5', '10', '20']
    markers = ['o', 'd', 's', '^']
    plt.figure(figsize=(10, 8))
    for df, color, label, marker in zip(dfs, colors, labels, markers):
        if len(mapped_average_number) == len(df.iloc[:, col]):
            plt.plot(mapped_average_number, df.iloc[:, col], label=label, color=color, marker=marker, markersize=10)
        else:
            print(f"Skipping {label} due to length mismatch: {len(mapped_average_number)} vs {len(df.iloc[:, col])}")
    plt.xlabel('t-step', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(list(mapping.values()), list(mapping.keys()))
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.savefig(filename)
    plt.close()

dfs = [naive_df, pian_df, pia_df, secmi_df]
plot_data(mapped_average_number, dfs, 6, 'AUC', 'plot_auc.png')
plot_data(mapped_average_number, dfs, 7, 'ASR', 'plot_asr.png')
plot_data(mapped_average_number, dfs, 8, 'TPR_FPR', 'plot_f1.png')
