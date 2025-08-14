import os
import numpy as np
import xarray as xr
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 路径
tccon_dir = 'ELYSIA/tccon'
cams_path = 'ELYSIA/cams-EGG4/data_sfc_17_20.nc'

# 读取CAMS数据
print("正在读取CAMS数据...")
cams_ds = xr.open_dataset(cams_path)
if 'valid_time' in cams_ds.dims:
    cams_ds = cams_ds.rename({'valid_time': 'time'})
if 'valid_time' in cams_ds.coords:
    cams_ds = cams_ds.rename({'valid_time': 'time'})
cams_xco2 = cams_ds['tcco2']

tccon_x, cams_x = [], []

print("正在处理TCCON数据...")
for file in glob.glob(os.path.join(tccon_dir, '*.nc')):
    ds = xr.open_dataset(file)
    # 假设变量名为xco2、lat、lon、time，实际请根据文件内容调整
    tccon_xco2 = ds['xco2'].values
    tccon_lat = ds['lat'].values
    tccon_lon = ds['long'].values
    tccon_time = ds['time'].values

    for i in tqdm(range(len(tccon_xco2)), desc=os.path.basename(file)):
        # 插值/提取CAMS
        cams_val = cams_xco2.sel(
            time=tccon_time[i], 
            latitude=tccon_lat[i], 
            longitude=tccon_lon[i], 
            method='nearest'
        ).values
        
        # 只添加有效的数据点
        if not np.isnan(tccon_xco2[i]) and not np.isnan(cams_val):
            tccon_x.append(tccon_xco2[i])
            cams_x.append(cams_val)

# 转为数组
tccon_x = np.array(tccon_x)
cams_x = np.array(cams_x)

print(f"总有效数据点数量: {len(tccon_x)}")

# 随机抽取100000个点
target_points = 100000
if len(tccon_x) > target_points:
    print(f"正在随机抽取 {target_points:,} 个数据点...")
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    indices = np.random.choice(len(tccon_x), target_points, replace=False)
    tccon_x = tccon_x[indices]
    cams_x = cams_x[indices]
    print(f"已抽取 {len(tccon_x):,} 个数据点")
else:
    print(f"数据点数量不足 {target_points:,}，使用全部 {len(tccon_x):,} 个数据点")

# 计算统计量
print("正在计算统计指标...")
r2 = r2_score(tccon_x, cams_x)
rmse = np.sqrt(mean_squared_error(tccon_x, cams_x))
mae = mean_absolute_error(tccon_x, cams_x)
bias = np.mean(cams_x - tccon_x)

print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f} ppm")
print(f"MAE = {mae:.4f} ppm")
print(f"Bias = {bias:.4f} ppm")

# 创建图形
print("正在创建图形...")
fig, ax = plt.subplots(figsize=(12, 10))

# 设置颜色映射（密度散点图）
# 计算点密度
print("正在计算点密度...")
xy = np.vstack([tccon_x, cams_x])
z = stats.gaussian_kde(xy)(xy)

# 绘制散点图，分批绘制以显示进度
print("正在绘制散点图...")
batch_size = 5000  # 每批绘制5000个点
n_batches = (len(tccon_x) + batch_size - 1) // batch_size

for i in tqdm(range(n_batches), desc="绘制散点图", unit="批次"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(tccon_x))
    
    if i == 0:
        # 第一批点
        scatter = ax.scatter(tccon_x[start_idx:end_idx], cams_x[start_idx:end_idx], 
                           c=z[start_idx:end_idx], cmap='viridis', alpha=0.6, 
                           s=20, edgecolors='white', linewidth=0.3)
    else:
        # 后续批次添加到同一个散点图对象
        ax.scatter(tccon_x[start_idx:end_idx], cams_x[start_idx:end_idx], 
                  c=z[start_idx:end_idx], cmap='viridis', alpha=0.6, 
                  s=20, edgecolors='white', linewidth=0.3)

print("正在添加颜色条...")
# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Point Density', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# 添加1:1线
print("正在添加1:1线和网格...")
x_min, x_max = tccon_x.min(), tccon_x.max()
y_min, y_max = cams_x.min(), cams_x.max()
plot_min = min(x_min, y_min)
plot_max = max(x_max, y_max)
ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, alpha=0.8, label='1:1 Line')

# 设置坐标轴
ax.set_xlim(plot_min - 1, plot_max + 1)
ax.set_ylim(plot_min - 1, plot_max + 1)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 设置标签和标题
print("正在设置标签和标题...")
ax.set_xlabel('TCCON XCO2 [ppm]', fontsize=14, fontweight='bold')
ax.set_ylabel('CAMS XCO2 [ppm]', fontsize=14, fontweight='bold')
ax.set_title('CAMS vs TCCON XCO2', 
            fontsize=16, fontweight='bold', pad=20)

# 添加统计信息
stats_text = f"""Statistics:
R² = {r2:.4f}
RMSE = {rmse:.2f} ppm
MAE = {mae:.2f} ppm
Bias = {bias:.2f} ppm"""

# 在图上添加统计信息框
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        verticalalignment='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

# 添加图例
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

# 设置刻度标签
ax.tick_params(axis='both', which='major', labelsize=12)

# 调整布局
print("正在调整布局...")
plt.tight_layout()

# 保存图片
print("正在保存图片...")
plt.savefig('CAMS_vs_TCCON_Real_Data_100k.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('CAMS_vs_TCCON_Real_Data_100k.pdf', dpi=300, bbox_inches='tight')

print("图片已保存为 'CAMS_vs_TCCON_Real_Data_100k.png' 和 'CAMS_vs_TCCON_Real_Data_100k.pdf'")

# 显示图片
print("正在显示图片...")
plt.show()

# 打印最终统计信息
print("\n" + "="*50)
print("最终统计结果:")
print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f} ppm")
print(f"MAE = {mae:.4f} ppm")
print(f"Bias = {bias:.4f} ppm")
print(f"数据点数量: {len(tccon_x):,}")
print("="*50)