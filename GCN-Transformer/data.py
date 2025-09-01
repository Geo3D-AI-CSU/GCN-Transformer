# -*- coding: utf-8 -*-
"""
功能：
  1. 加载 CAMS-EGG4、CAMS-IO、ERA5（wind）以及 OCO-2 卫星观测数据
  2. 过滤 OCO-2 的质量标记并按时间窗网格化到 CAMS 网格
  3. 将 ERA5/CAMS-IO 插值到 CAMS-EGG4 的时间与空间网格
  4. 计算风速，合并为一个统一的 Dataset
  5. 分块计算标准化与归一化参数并应用
  6. 构建 GCN + Transformer（时序图输入）的 PyTorch Geometric 数据样本并保存
"""

# ------------------ 导入模块 ------------------
import xarray as xr            # xarray：用于读取和处理多维数组（NetCDF 等）
import numpy as np             # numpy：数值计算基础库
import pandas as pd            # pandas：时间序列与表格数据处理
import glob                    # glob：文件路径通配符匹配
import os                      # os：操作系统交互（路径、目录等）
from tqdm import tqdm          # tqdm：循环进度条显示
from sklearn.preprocessing import StandardScaler  # Sklearn 的标准化工具（未直接使用，但保留）
import torch                   # PyTorch：张量与深度学习基础
from scipy.spatial import cKDTree  # cKDTree：高效的最近邻/球邻域查询

# 新增：PyTorch Geometric 数据结构
from torch_geometric.data import Data, InMemoryDataset  # Data（图的基本容器），InMemoryDataset（内存型数据集模板）


# ------------------ 辅助函数：构建网格邻接 ------------------
def build_grid_edge_index(H, W):
    """构建规则网格的 4 邻接边索引，返回形状为 [2, num_edges] 的 long Tensor。
    参数：
        H: 网格的行数（纬度数）
        W: 网格的列数（经度数）
    说明：节点按行优先编号：idx = i * W + j
    返回：edge_index（torch.LongTensor），可直接用于 PyG 的 Data.edge_index
    """
    edge_list = []                        # 用 Python 列表收集边对 (source, target)
    for i in range(H):                    # 遍历每一行（纬度索引）
        for j in range(W):                # 遍历每一列（经度索引）
            idx = i * W + j               # 计算当前节点的线性索引
            # 上下左右四个方向的偏移
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj      # 计算邻居的二维坐标
                if 0<=ni<H and 0<=nj<W:   # 检查邻居是否在网格内
                    nidx = ni*W + nj    # 计算邻居线性索引
                    edge_list.append([idx, nidx])  # 添加一条有向边 idx->nidx
    edge_index = np.array(edge_list).T    # 转为 numpy 数组并转置为 [2, E]
    return torch.tensor(edge_index, dtype=torch.long)  # 转为 PyTorch tensor 并返回


# ------------------ 主处理函数 ------------------
def process_and_save_data():
    print("开始数据处理...")  # 打印流程开始提示

    # 获取当前脚本所在目录并创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本绝对路径的目录
    output_dir = os.path.join(current_dir, "processed_data") # 输出数据存放目录
    os.makedirs(output_dir, exist_ok=True)                    # 若目录不存在则创建（exist_ok=True 表示若已存在则不报错）

    # ==== Step 1: 加载数据 ====
    print("Step 1: 加载数据...")
    cams_egg4_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-EGG4", "data_sfc_17_20.nc")))
    # 使用 glob 找到 cams-egg4 的文件列表并排序（路径可为通配形式，这里示例为单文件）

    cams_egg4 = xr.open_mfdataset(cams_egg4_files, combine='by_coords', chunks={'valid_time': 100})
    # 使用 xarray.open_mfdataset 批量打开并按坐标合并，指定 dask 块大小（chunks）以支持延迟加载

    cams_egg4 = cams_egg4.rename({'valid_time': 'time'})  # 将变量名 valid_time 统一重命名为 time，便于对齐

    cams_io_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-IO", "cams73_latest_co2_col_surface_inst_*.nc")))
    # 匹配 CAMS-IO 的多个文件（通配匹配）

    cams_io = xr.open_mfdataset(cams_io_files, combine='by_coords', chunks={'time': 100})
    # 打开并合并 CAMS-IO 数据集

    wind_files = sorted(glob.glob(os.path.join("ELYSIA", "ERA-5", "ERA5_*.nc")))
    # 匹配 ERA5 风场文件（可能为多个小文件）

    wind = xr.open_mfdataset(wind_files, combine='by_coords', chunks={'time': 100})
    # 打开并合并 ERA5 数据集

    # 辅助函数：把 Dataset 中的 float64 变量转换为 float32，以减少内存占用
    def ensure_float32(ds):
        for var in ds.data_vars:                      # 遍历数据集中的数据变量
            if ds[var].dtype in [np.float64]:         # 如果变量为 float64
                ds[var] = ds[var].astype(np.float32) # 转为 float32
        return ds

    # 应用精度转换以节约内存
    cams_egg4 = ensure_float32(cams_egg4)
    cams_io = ensure_float32(cams_io)
    wind = ensure_float32(wind)

    # ==== Step 2: 处理 OCO-2 数据（逐文件读取并过滤质量标记）====
    print("Step 2: 处理OCO-2数据...")

    oco2_files = sorted(glob.glob(os.path.join("ELYSIA", "oco-2", "oco2_LtCO2_*.nc4")))
    # 匹配 OCO-2 L2 文件（nc4 格式）

    oco2_filtered_list = []  # 用来收集每个文件过滤后的 Dataset
    for file in tqdm(oco2_files, desc="筛选OCO-2质量数据"):  # 遍历所有匹配到的文件，并显示进度条
        ds = xr.open_dataset(file)  # 打开单个 OCO-2 文件（通常为较小的二维/一维数组）
        # 只保留感兴趣的变量以节省内存：sounding_id、latitude、longitude、xco2、xco2_quality_flag
        vars_needed = [v for v in ['sounding_id', 'latitude', 'longitude', 'xco2', 'xco2_quality_flag'] if v in ds.variables]
        ds = ds[vars_needed]  # 按需选择变量

        # 若 xco2 存在且为 float（非 float32），则转换为 float32
        if 'xco2' in ds.variables and np.issubdtype(ds['xco2'].dtype, np.floating) and ds['xco2'].dtype != np.float32:
            ds['xco2'] = ds['xco2'].astype(np.float32)

        good_flag = ds['xco2_quality_flag'] == 0  # 质量标记等于 0 表示观测质量合格
        ds_good = ds.where(good_flag, drop=True)  # 仅保留质量合格的记录（drop=True 表示删除不符合条件的索引）
        oco2_filtered_list.append(ds_good)       # 将过滤后的 Dataset 加入列表

    # 把所有文件按 sounding_id 维拼接在一起，形成一个大的 OCO-2 Dataset
    oco2_filtered = xr.concat(oco2_filtered_list, dim='sounding_id')

    # 从 sounding_id 中解析时间：有些 OCO-2 数据的 sounding_id 前 8 位为 YYYYMMDD
    sounding_id = oco2_filtered['sounding_id'].values
    sounding_id_str = sounding_id.astype(np.int64).astype(str)                       # 转为字符串以便切片
    dates = pd.to_datetime([s[:8] for s in sounding_id_str], format='%Y%m%d')        # 解析为 pandas 的 datetime

    # 将解析出的日期作为新的 time 坐标，和 sounding_id 对齐
    oco2_filtered = oco2_filtered.assign_coords(time=("sounding_id", dates))

    # 检查原始 OCO-2 xco2 的 NaN 比例，作为数据质量的粗略提示
    nan_ratio_oco2 = np.isnan(oco2_filtered['xco2'].values).mean()
    print(f"原始OCO-2观测xco2的NaN比例: {nan_ratio_oco2:.4f}")

    # ==== Step 3: 网格化 OCO-2 数据，将稀疏观测映射到 CAMS 网格上 ====
    print("Step 3: 网格化OCO-2数据...")

    # 注意：这里使用 meshgrid 生成的顺序为 latitude × longitude（indexing='ij'）以匹配 xarray 常用顺序
    lat_grid, lon_grid = np.meshgrid(cams_egg4.latitude, cams_egg4.longitude, indexing='ij')
    grid_coords = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))  # 把 2D 网格展平为 N×2 的坐标数组

    gridded_oco2_list = []  # 存放每个时间步的网格化结果
    for i_t, t in enumerate(tqdm(cams_egg4.time.values, desc="网格化OCO-2")):  # 遍历目标时间轴（CAMS 的时间点）
        # 以目标时间 t 为中心，取前后 180 分钟（3 小时）作为时间窗
        t1 = np.datetime64(t) - np.timedelta64(180, 'm')
        t2 = np.datetime64(t) + np.timedelta64(180, 'm')

        # 选取 time 坐标在 [t1, t2] 时间窗内的 OCO-2 观测
        subset = oco2_filtered.sel(
            sounding_id=((oco2_filtered['time'] >= t1) & (oco2_filtered['time'] <= t2))
        )

        # 若该时间窗口没有任何观测，直接填充全 NaN 的网格并跳过
        if subset.sounding_id.size == 0:
            gridded_oco2_list.append(np.full(lat_grid.shape, np.nan, dtype=np.float32))
            continue

        # 把观测点的经纬度拼成 M×2 的数组供 KDTree 使用
        obs_coords = np.column_stack((subset.latitude.values, subset.longitude.values))
        tree = cKDTree(obs_coords)  # 构建 KD 树用于空间邻域查询

        radius = 0.5  # 搜索半径（单位为度），注意：这是经纬度的度数近似，非真正的球面距离
        indices_list = tree.query_ball_point(grid_coords, r=radius)  # 对每个网格点查询邻域内的观测索引列表

        xco2_values = subset['xco2'].values  # 观测值数组
        grid_flat = np.full(grid_coords.shape[0], np.nan, dtype=np.float32)  # 一维展平网格（用于临时填充）

        # 对每个网格点：如果邻域内观测点数量 >= 3，则取邻域均值，否则保持 NaN
        for i, indices in enumerate(indices_list):
            if len(indices) >= 3:
                grid_flat[i] = np.nanmean(xco2_values[indices])

        grid = grid_flat.reshape(lat_grid.shape)  # 恢复为二维网格 (lat, lon)
        gridded_oco2_list.append(grid)            # 收集该时间步结果

    # 将所有时间步堆叠为一个 DataArray，维度为 [time, latitude, longitude]
    oco2_gridded = xr.DataArray(
        data=np.stack(gridded_oco2_list),
        coords={
            'time': cams_egg4.time.values,
            'latitude': cams_egg4.latitude,
            'longitude': cams_egg4.longitude
        },
        dims=['time', 'latitude', 'longitude'],
        name='xco2_gridded'
    )

    # 打印网格化后 oco2 的 NaN 比例，便于观察网格覆盖率
    nan_ratio_gridded = np.isnan(oco2_gridded.values).mean()
    print(f"网格化后OCO-2 xco2的NaN比例: {nan_ratio_gridded:.4f}")

    # ==== Step 4: 插值辅助数据（把 wind、cams_io 插值到 cams_egg4 的时间和空间网格）====
    print("Step 4: 插值辅助数据...")
    if 'valid_time' in wind.coords:                   # 有些 ERA5 文件的时间维名为 valid_time
        wind = wind.rename({'valid_time': 'time'})    # 统一改为 time

    # 定义分月插值函数，避免一次性插值造成巨大内存峰值
    def interp_in_month_chunks(ds, target_times, latitudes, longitudes):
        result_list = []
        times = pd.to_datetime(target_times)                        # 把目标时间数组转为 pandas datetime
        months = sorted(set([(t.year, t.month) for t in times]))    # 构建 (year, month) 唯一列表
        for year, month in months:                                  # 按月循环
            mask = (times.year == year) & (times.month == month)    # 掩码选择该月的时间索引
            t_chunk = target_times[mask]                            # 该月的目标时间
            if len(t_chunk) == 0:
                continue
            print(f"插值: {year}-{month:02d} 共{len(t_chunk)}步")
            chunk_interp = ds.interp(
                time=t_chunk,
                latitude=latitudes,
                longitude=longitudes
            )
            result_list.append(chunk_interp)
        return xr.concat(result_list, dim='time')  # 将每个月的插值结果沿时间维拼接

    # 对 wind（ERA5）与 cams_io 分别按月插值到 cams_egg4 的时间与空间网格
    wind_interp = interp_in_month_chunks(
        wind, cams_egg4.time.values, cams_egg4.latitude, cams_egg4.longitude
    )
    cams_io_interp = interp_in_month_chunks(
        cams_io, cams_egg4.time.values, cams_egg4.latitude, cams_egg4.longitude
    )

    # ==== Step 5: 合并数据集 ====
    print("Step 5: 合并数据...")
    wind_speed = (wind_interp['u10']**2 + wind_interp['v10']**2)**0.5    # 计算 10m 风速模长：sqrt(u^2 + v^2)
    wind_speed = wind_speed.astype(np.float32)                         # 转为 float32

    # 合并为一个 xarray.Dataset，包含 EGG4、IO、风速、风分量以及网格化的 OCO-2
    combined = xr.Dataset({
        'cams_egg4': cams_egg4['tcco2'] if 'tcco2' in cams_egg4 else cams_egg4['XCO2'],
        'cams_io': cams_io_interp['XCO2'],
        'wind_u': wind_interp['u10'],
        'wind_v': wind_interp['v10'],
        'wind_speed': wind_speed,
        'oco2_xco2': oco2_gridded
    })

    # ==== Step 6: 标准化与归一化（分块计算统计量以减小内存占用）====
    print("combined['cams_egg4'] shape:", combined['cams_egg4'].shape)
    print("combined.dims:", combined.dims)

    input_vars = ['cams_egg4', 'cams_io', 'wind_u', 'wind_v', 'wind_speed']  # 待处理变量清单

    norm_params = {}    # 存放每个变量的 mean/std
    minmax_params = {}  # 存放每个变量的 min/max

    # 逐变量按时间分块计算总和、平方和、最小值与最大值，从而得到全局 mean/std/min/max
    for var in input_vars:
        print(f"计算 {var} 的统计参数...")
        data_array = combined[var]                  # 获取 xarray.DataArray

        total_sum = 0.0                             # 累计和
        total_count = 0                             # 有效值计数（非 NaN）
        total_sum_sq = 0.0                          # 累计平方和
        min_val = float('inf')                      # 全局最小值初始化
        max_val = float('-inf')                     # 全局最大值初始化

        chunk_size = 30                             # 每次处理的时间块长度，避免一次性内存峰值
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 统计参数"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))  # 按时间切片取块
            chunk_values = chunk.values                        # 转为 numpy 数组
            valid_mask = ~np.isnan(chunk_values)               # 非 NaN 掩码
            valid_data = chunk_values[valid_mask]              # 压缩为 1D 的有效数据

            if len(valid_data) > 0:
                valid_data = valid_data.astype(np.float32)    # 转为 float32 加速计算并节省内存
                total_sum += np.sum(valid_data)               # 更新总和
                total_count += len(valid_data)                # 更新计数
                total_sum_sq += np.sum(valid_data ** 2)       # 更新平方和
                min_val = min(min_val, np.min(valid_data))   # 更新最小值
                max_val = max(max_val, np.max(valid_data))   # 更新最大值

        # 在块处理完成后计算均值与标准差（若无有效值则设置默认 mean=0, std=1）
        if total_count > 0:
            mean = float(total_sum / total_count)
            variance = float((total_sum_sq / total_count) - (mean ** 2))
            std = float(np.sqrt(variance) if variance > 0 else 1.0)
        else:
            mean = 0.0
            std = 1.0

        norm_params[var] = {"mean": mean, "std": std}   # 记录标准化参数
        minmax_params[var] = {"min": float(min_val if min_val != float('inf') else 0),
                              "max": float(max_val if max_val != float('-inf') else 1)}  # 记录极值

    # 分块应用标准化与归一化
    combined_norm = {}  # 用字典保存每个变量的归一化结果（xarray.DataArray）
    for var in input_vars:
        print(f"标准化和归一化 {var}...")
        data_array = combined[var]

        normalized_chunks = []
        chunk_size = 30
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 标准化归一化"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))  # 取时间块
            chunk_values = chunk.values.astype(np.float32)   # 降精度以节省内存

            # 标准化
            data_norm = (chunk_values - norm_params[var]['mean']) / norm_params[var]['std']
            # 归一化
            data_minmax = (data_norm - minmax_params[var]['min']) / (minmax_params[var]['max'] - minmax_params[var]['min'])

            normalized_chunks.append(data_minmax.astype(np.float32))  # 保存当前块的处理结果

        normalized_data = np.concatenate(normalized_chunks, axis=0).astype(np.float32)  # 将所有时间块沿 time 轴拼接
        normalized_data = normalized_data.reshape(data_array.shape)  # 恢复到原始的 shape，保证 dims 一致

        # 用 xarray.DataArray 保存，保持 dims 和 coords，以便后续按坐标访问
        combined_norm[var] = xr.DataArray(
            normalized_data,
            dims=data_array.dims,
            coords=data_array.coords
        )

    # ==== Step 7: 构建 GCN + Transformer 所需的图与时序输入 ====
    print("Step 7: 构建GCN+Transformer输入...")

    seq_len = 3  # 时间序列长度（历史帧数）
    T = combined['cams_egg4'].shape[0]  # 时间轴长度
    H = combined['cams_egg4'].shape[1]  # 纬度格点数（行数）
    W = combined['cams_egg4'].shape[2]  # 经度格点数（列数）
    C = len(input_vars)                 # 通道数（特征变量数）

    # 把每个变量的归一化数据按通道堆叠为一个 4D numpy 数组，维度顺序为 [T, H, W, C]
    data_all = np.stack([combined_norm[var].values for var in input_vars], axis=-1)  # shape: [T, H, W, C]
    target = combined['oco2_xco2'].values.astype(np.float32)  # 目标标签，shape: [T, H, W]

    # 构建规则网格的边索引，用于 GCN 的空间拓扑
    edge_index = build_grid_edge_index(H, W)  # torch.LongTensor，shape: [2, E]

    # 按时间步构建样本列表，每个样本为一个 Data 对象，x 为时序特征，y 为当前时间的标签
    samples = []
    for t in tqdm(range(seq_len, T), desc="构建图样本"):
        x_seq = []
        # 取 seq_len 个历史时间帧作为输入（不包含目标时刻 t）
        for dt in range(t-seq_len, t):
            x_t = data_all[dt]  # 取出某一时间帧的 [H, W, C] 数据

            # 若该帧存在 NaN（缺失），使用该帧的全局均值填充
            if np.any(np.isnan(x_t)):
                x_t = np.nan_to_num(x_t, nan=np.nanmean(x_t))

            # 将 2D 网格展开为节点向量，shape: [num_nodes, C]
            x_seq.append(x_t.reshape(-1, C))

        # 目标标签：目标时间 t 的网格化 OCO-2，展开为一维向量
        y_t = target[t].reshape(-1)
        if np.any(np.isnan(y_t)):
            y_t = np.nan_to_num(y_t, nan=np.nanmean(y_t))  # 使用标签的全局均值填充缺失值

        # 将历史帧堆叠为 [seq_len, num_nodes, C]
        x_seq = np.stack(x_seq, axis=0)

        # 构造 PyG 的 Data 对象；注意：这里把 x 放为整段时序（可在模型内自行处理时序维）
        data = Data(x=torch.tensor(x_seq, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(y_t, dtype=torch.float))
        samples.append(data)  # 收集样本

    # ===== 新增：按时间顺序排序一次（防止顺序被破坏） =====
    samples = sorted(samples, key=lambda s: s.x.shape[0])
    # =============================================

    # 保存所有样本为单一 .pt 文件（注意：若样本量大，文件可能很大）
    torch.save(samples, os.path.join(output_dir, "gcn_transformer_data.pt"))
    print(f"数据处理完成！共保存 {len(samples)} 个图样本，数据已保存到: {output_dir}")


if __name__ == "__main__":
    process_and_save_data()  # 作为脚本直接运行时执行主流程
