import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from scipy.spatial import cKDTree

# 新增：PyTorch Geometric数据结构
from torch_geometric.data import Data, InMemoryDataset


def build_grid_edge_index(H, W):
    """构建规则网格的4邻接边index，返回[2, num_edges]"""
    edge_list = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            # 上下左右
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0<=ni<H and 0<=nj<W:
                    nidx = ni*W + nj
                    edge_list.append([idx, nidx])
    edge_index = np.array(edge_list).T  # [2, num_edges]
    return torch.tensor(edge_index, dtype=torch.long)


def process_and_save_data():
    print("开始数据处理...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    # ==== Step 1: 加载数据 ====
    print("Step 1: 加载数据...")
    cams_egg4_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-EGG4", "data_sfc_17_20.nc")))
    cams_egg4 = xr.open_mfdataset(cams_egg4_files, combine='by_coords', chunks={'valid_time': 100})
    cams_egg4 = cams_egg4.rename({'valid_time': 'time'})  
    cams_io_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-IO", "cams73_latest_co2_col_surface_inst_*.nc")))
    cams_io = xr.open_mfdataset(cams_io_files, combine='by_coords', chunks={'time': 100})
    wind_files = sorted(glob.glob(os.path.join("ELYSIA", "ERA-5", "ERA5_*.nc")))
    wind = xr.open_mfdataset(wind_files, combine='by_coords', chunks={'time': 100})

    def ensure_float32(ds):
        for var in ds.data_vars:
            if ds[var].dtype in [np.float64]:
                ds[var] = ds[var].astype(np.float32)
        return ds
    cams_egg4 = ensure_float32(cams_egg4)
    cams_io = ensure_float32(cams_io)
    wind = ensure_float32(wind)

    # ==== Step 2: 处理OCO-2数据 ====
    print("Step 2: 处理OCO-2数据...")
    oco2_files = sorted(glob.glob(os.path.join("ELYSIA", "oco-2", "oco2_LtCO2_*.nc4")))
    oco2_filtered_list = []
    for file in tqdm(oco2_files, desc="筛选OCO-2质量数据"):
        ds = xr.open_dataset(file)
        vars_needed = [v for v in ['sounding_id', 'latitude', 'longitude', 'xco2', 'xco2_quality_flag'] if v in ds.variables]
        ds = ds[vars_needed]
        if 'xco2' in ds.variables and np.issubdtype(ds['xco2'].dtype, np.floating) and ds['xco2'].dtype != np.float32:
            ds['xco2'] = ds['xco2'].astype(np.float32)
        good_flag = ds['xco2_quality_flag'] == 0
        ds_good = ds.where(good_flag, drop=True)
        oco2_filtered_list.append(ds_good)
    oco2_filtered = xr.concat(oco2_filtered_list, dim='sounding_id')
    sounding_id = oco2_filtered['sounding_id'].values
    sounding_id_str = sounding_id.astype(np.int64).astype(str)
    dates = pd.to_datetime([s[:8] for s in sounding_id_str], format='%Y%m%d')
    oco2_filtered = oco2_filtered.assign_coords(time=("sounding_id", dates))

    # 检查原始OCO-2 xco2的NaN比例
    nan_ratio_oco2 = np.isnan(oco2_filtered['xco2'].values).mean()
    print(f"原始OCO-2观测xco2的NaN比例: {nan_ratio_oco2:.4f}")

    # ==== Step 3: 网格化OCO-2数据 ====
    print("Step 3: 网格化OCO-2数据...")
    lat_grid, lon_grid = np.meshgrid(cams_egg4.latitude, cams_egg4.longitude, indexing='ij')
    grid_coords = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    gridded_oco2_list = []
    for i_t, t in enumerate(tqdm(cams_egg4.time.values, desc="网格化OCO-2")):
        t1 = np.datetime64(t) - np.timedelta64(180, 'm')
        t2 = np.datetime64(t) + np.timedelta64(180, 'm')
        subset = oco2_filtered.sel(
            sounding_id=((oco2_filtered['time'] >= t1) & (oco2_filtered['time'] <= t2))
        )
        if subset.sounding_id.size == 0:
            gridded_oco2_list.append(np.full(lat_grid.shape, np.nan, dtype=np.float32))
            continue
        obs_coords = np.column_stack((subset.latitude.values, subset.longitude.values))
        tree = cKDTree(obs_coords)
        radius = 0.5  
        indices_list = tree.query_ball_point(grid_coords, r=radius)
        xco2_values = subset['xco2'].values
        grid_flat = np.full(grid_coords.shape[0], np.nan, dtype=np.float32)
        for i, indices in enumerate(indices_list):
            if len(indices) >= 3:  
                grid_flat[i] = np.nanmean(xco2_values[indices])
        grid = grid_flat.reshape(lat_grid.shape)
        gridded_oco2_list.append(grid)
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

    # 打印网格化后oco2_gridded的NaN比例
    nan_ratio_gridded = np.isnan(oco2_gridded.values).mean()
    print(f"网格化后OCO-2 xco2的NaN比例: {nan_ratio_gridded:.4f}")

    # ==== Step 4: 插值辅助数据 ====
    print("Step 4: 插值辅助数据...")
    if 'valid_time' in wind.coords:
        wind = wind.rename({'valid_time': 'time'})
    def interp_in_month_chunks(ds, target_times, latitudes, longitudes):
        result_list = []
        times = pd.to_datetime(target_times)
        months = sorted(set([(t.year, t.month) for t in times]))
        for year, month in months:
            mask = (times.year == year) & (times.month == month)
            t_chunk = target_times[mask]
            if len(t_chunk) == 0:
                continue
            print(f"插值: {year}-{month:02d} 共{len(t_chunk)}步")
            chunk_interp = ds.interp(
                time=t_chunk,
                latitude=latitudes,
                longitude=longitudes
            )
            result_list.append(chunk_interp)
        return xr.concat(result_list, dim='time')
    wind_interp = interp_in_month_chunks(
        wind, cams_egg4.time.values, cams_egg4.latitude, cams_egg4.longitude
    )
    cams_io_interp = interp_in_month_chunks(
        cams_io, cams_egg4.time.values, cams_egg4.latitude, cams_egg4.longitude
    )

    # ==== Step 5: 合并数据 ====
    print("Step 5: 合并数据...")
    wind_speed = (wind_interp['u10']**2 + wind_interp['v10']**2)**0.5
    wind_speed = wind_speed.astype(np.float32)
    combined = xr.Dataset({
        'cams_egg4': cams_egg4['tcco2'] if 'tcco2' in cams_egg4 else cams_egg4['XCO2'],
        'cams_io': cams_io_interp['XCO2'],
        'wind_u': wind_interp['u10'],
        'wind_v': wind_interp['v10'],
        'wind_speed': wind_speed,
        'oco2_xco2': oco2_gridded
    })

    # ==== Step 6: 标准化和归一化 ====
    print("combined['cams_egg4'] shape:", combined['cams_egg4'].shape)
    print("combined.dims:", combined.dims)
    input_vars = ['cams_egg4', 'cams_io', 'wind_u', 'wind_v', 'wind_speed']
    norm_params = {}
    minmax_params = {}
    for var in input_vars:
        print(f"计算 {var} 的统计参数...")
        data_array = combined[var]
        total_sum = 0.0
        total_count = 0
        total_sum_sq = 0.0
        min_val = float('inf')
        max_val = float('-inf')
        chunk_size = 30
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 统计参数"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))
            chunk_values = chunk.values
            valid_mask = ~np.isnan(chunk_values)
            valid_data = chunk_values[valid_mask]
            if len(valid_data) > 0:
                valid_data = valid_data.astype(np.float32)
                total_sum += np.sum(valid_data)
                total_count += len(valid_data)
                total_sum_sq += np.sum(valid_data ** 2)
                min_val = min(min_val, np.min(valid_data))
                max_val = max(max_val, np.max(valid_data))
        if total_count > 0:
            mean = float(total_sum / total_count)
            variance = float((total_sum_sq / total_count) - (mean ** 2))
            std = float(np.sqrt(variance) if variance > 0 else 1.0)
        else:
            mean = 0.0
            std = 1.0
        norm_params[var] = {"mean": mean, "std": std}
        minmax_params[var] = {"min": float(min_val if min_val != float('inf') else 0), "max": float(max_val if max_val != float('-inf') else 1)}
    # 分块标准化归一化
    combined_norm = {}
    for var in input_vars:
        print(f"标准化和归一化 {var}...")
        data_array = combined[var]
        
        normalized_chunks = []
        chunk_size = 30
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 标准化归一化"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))
            chunk_values = chunk.values.astype(np.float32)
            data_norm = (chunk_values - norm_params[var]['mean']) / norm_params[var]['std']
            data_minmax = (data_norm - minmax_params[var]['min']) / (minmax_params[var]['max'] - minmax_params[var]['min'])
            normalized_chunks.append(data_minmax.astype(np.float32))
        normalized_data = np.concatenate(normalized_chunks, axis=0).astype(np.float32)
        normalized_data = normalized_data.reshape(data_array.shape)  # 保证shape一致
        # 用xarray.DataArray保存，保留dims和coords
        combined_norm[var] = xr.DataArray(
            normalized_data,
            dims=data_array.dims,
            coords=data_array.coords
        )

    # ==== Step 7: 构建GCN+Transformer输入 ====
    print("Step 7: 构建GCN+Transformer输入...")
    seq_len = 3
    T = combined['cams_egg4'].shape[0]
    H = combined['cams_egg4'].shape[1]
    W = combined['cams_egg4'].shape[2]
    C = len(input_vars)
    data_all = np.stack([combined_norm[var].values for var in input_vars], axis=-1)  # [T, H, W, C]
    target = combined['oco2_xco2'].values.astype(np.float32)  # [T, H, W]
    edge_index = build_grid_edge_index(H, W)

    samples = []
    for t in tqdm(range(seq_len, T), desc="构建图样本"):
        x_seq = []
        for dt in range(t-seq_len, t):
            x_t = data_all[dt]
            if np.any(np.isnan(x_t)):
                x_t = np.nan_to_num(x_t, nan=np.nanmean(x_t))
            x_seq.append(x_t.reshape(-1, C))

        y_t = target[t].reshape(-1)
        if np.any(np.isnan(y_t)):
            y_t = np.nan_to_num(y_t, nan=np.nanmean(y_t))

        x_seq = np.stack(x_seq, axis=0)  # [seq_len, num_nodes, C]
        data = Data(x=torch.tensor(x_seq, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(y_t, dtype=torch.float))
        samples.append(data)

    # ===== 新增：按时间顺序排序一次（防止顺序被破坏） =====
    samples = sorted(samples, key=lambda s: s.x.shape[0])  # 实际这里 shape[0] 对应时间顺序
    # =============================================

    torch.save(samples, os.path.join(output_dir, "gcn_transformer_data.pt"))
    print(f"数据处理完成！共保存 {len(samples)} 个图样本，数据已保存到: {output_dir}")
if __name__ == "__main__":
    process_and_save_data()