import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
from scipy.spatial import cKDTree

def process_and_save_data():
    """处理数据并保存到文件"""
    print("开始数据处理...")
    
    # 创建保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # ==== Step 1: 加载数据 ====
    print("Step 1: 加载数据...")
    cams_egg4_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-EGG4", "data_sfc_17_20.nc")))
    cams_egg4 = xr.open_mfdataset(cams_egg4_files, combine='by_coords', chunks={'valid_time': 100})
    # 将valid_time重命名为time，以便与其他数据集保持一致
    cams_egg4 = cams_egg4.rename({'valid_time': 'time'})
    
    cams_io_files = sorted(glob.glob(os.path.join("ELYSIA", "cams-IO", "cams73_latest_co2_col_surface_inst_*.nc")))
    cams_io = xr.open_mfdataset(cams_io_files, combine='by_coords', chunks={'time': 100})
    
    wind_files = sorted(glob.glob(os.path.join("ELYSIA", "ERA-5", "ERA5_20*.nc")))
    wind = xr.open_mfdataset(wind_files, combine='by_coords', chunks={'time': 100})
    
    # 设置数据类型为float32以减少内存使用
    print("设置数据类型为float32...")
    
    # 确保所有数据集使用float32
    def ensure_float32(ds):
        """确保数据集中的所有数值变量使用float32"""
        for var in ds.data_vars:
            if ds[var].dtype in [np.float64]:
                ds[var] = ds[var].astype(np.float32)
        return ds
    
    # 应用float32转换
    cams_egg4 = ensure_float32(cams_egg4)
    cams_io = ensure_float32(cams_io)
    wind = ensure_float32(wind)
    
    # ==== Step 2: 处理OCO-2数据 ====
    print("Step 2: 处理OCO-2数据...")
    oco2_files = sorted(glob.glob(os.path.join("ELYSIA", "oco-2", "oco2_LtCO2_*.nc4")))
    oco2_filtered_list = []
    for file in tqdm(oco2_files, desc="筛选OCO-2质量数据"):
        ds = xr.open_dataset(file)
        # 只保留需要的变量，减少内存压力
        vars_needed = [v for v in ['sounding_id', 'latitude', 'longitude', 'xco2', 'xco2_quality_flag'] if v in ds.variables]
        ds = ds[vars_needed]
        # 确保xco2数据使用float32
        if 'xco2' in ds.variables and np.issubdtype(ds['xco2'].dtype, np.floating) and ds['xco2'].dtype != np.float32:
            ds['xco2'] = ds['xco2'].astype(np.float32)
        good_flag = ds['xco2_quality_flag'] == 0
        ds_good = ds.where(good_flag, drop=True)
        oco2_filtered_list.append(ds_good)
    oco2_filtered = xr.concat(oco2_filtered_list, dim='sounding_id')
    # 重新赋值oco2_filtered的时间坐标
    sounding_id = oco2_filtered['sounding_id'].values
    sounding_id_str = sounding_id.astype(np.int64).astype(str)
    dates = pd.to_datetime([s[:8] for s in sounding_id_str], format='%Y%m%d')
    oco2_filtered = oco2_filtered.assign_coords(time=("sounding_id", dates))
    
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
        
        radius = 0.375
        indices_list = tree.query_ball_point(grid_coords, r=radius)
        
        xco2_values = subset['xco2'].values
        grid_flat = np.full(grid_coords.shape[0], np.nan, dtype=np.float32)
        
        for i, indices in enumerate(indices_list):
            if len(indices) >= 10:
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
    
    # ==== Step 4: 插值辅助数据 ====
    print("Step 4: 插值辅助数据...")

    # 检查wind数据集的时间维度名称并统一为time
    if 'valid_time' in wind.coords:
        wind = wind.rename({'valid_time': 'time'})

    def interp_in_month_chunks(ds, target_times, latitudes, longitudes):
        """按月分块插值，减少内存压力"""
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
    # 确保风速计算使用float32
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
    
    # ==== Step 6: 整体标准化和归一化 ====
    print("Step 6: 整体标准化和归一化...")

    input_vars = ['cams_egg4', 'cams_io', 'wind_u', 'wind_v', 'wind_speed']

    # 分块计算均值、方差、最小值、最大值
    norm_params = {}
    minmax_params = {}
    
    for var in input_vars:
        print(f"计算 {var} 的统计参数...")
        data_array = combined[var]
        
        # 分块计算统计量，使用float32
        total_sum = 0.0
        total_count = 0
        total_sum_sq = 0.0
        min_val = float('inf')
        max_val = float('-inf')
        
        # 按时间分块处理
        chunk_size = 30 # 每次处理30个时间步
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 统计参数"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))
            chunk_values = chunk.values
            valid_mask = ~np.isnan(chunk_values)
            valid_data = chunk_values[valid_mask]
            
            if len(valid_data) > 0:
                # 转换为float32进行计算
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

    # 分块进行标准化和归一化
    combined_norm = combined.copy()
    for var in input_vars:
        print(f"标准化和归一化 {var}...")
        data_array = combined[var]
        
        # 分块处理标准化和归一化
        normalized_chunks = []
        chunk_size = 30      
        for i in tqdm(range(0, len(data_array.time), chunk_size), desc=f"{var} 标准化归一化"):
            end_idx = min(i + chunk_size, len(data_array.time))
            chunk = data_array.isel(time=slice(i, end_idx))
            chunk_values = chunk.values.astype(np.float32)
            
            # 标准化
            data_norm = (chunk_values - norm_params[var]['mean']) / norm_params[var]['std']
            # 归一化
            data_minmax = (data_norm - minmax_params[var]['min']) / (minmax_params[var]['max'] - minmax_params[var]['min'])
            
            normalized_chunks.append(data_minmax.astype(np.float32))
        
        # 合并所有块，使用float32减少内存
        normalized_data = np.concatenate(normalized_chunks, axis=0).astype(np.float32)
        combined_norm[var] = (data_array.dims, normalized_data)
    
    # ==== Step 7: 构建CNN和LSTM输入 ====
    print("Step 7: 构建CNN和LSTM输入...")
    data_cnn = combined_norm[input_vars].to_array().transpose("time", "latitude", "longitude", "variable")
    target = combined['oco2_xco2']
    
    def build_patches_and_sequences(data_cnn, cams_egg4, cams_io, target, patch_size=7, seq_len=3):
        pad = patch_size // 2
        T, H, W, C = data_cnn.shape
        X_cnn, X_lstm, y = [], [], []
        
        for t in range(seq_len, T):
            for i in range(pad, H - pad):
                for j in range(pad, W - pad):
                    patch = data_cnn[t, i - pad:i + pad + 1, j - pad:j + pad + 1, :]
                    if np.any(np.isnan(patch)):
                        continue
                        
                    seq_egg4 = cams_egg4[t - seq_len:t, i, j]
                    seq_io = cams_io[t - seq_len:t, i, j]
                    if np.any(np.isnan(seq_egg4)) or np.any(np.isnan(seq_io)):
                        continue
                        
                    label = target[t, i, j]
                    if np.isnan(label):
                        continue
                        
                    seq = np.stack([seq_egg4, seq_io], axis=-1)
                    X_cnn.append(patch)
                    X_lstm.append(seq)
                    y.append(label)
                    
        return np.array(X_cnn, dtype=np.float32), np.array(X_lstm, dtype=np.float32), np.array(y, dtype=np.float32)
    
    X_cnn, X_lstm, y = build_patches_and_sequences(
        data_cnn.values.astype(np.float32),
        combined_norm['cams_egg4'].values.astype(np.float32),
        combined_norm['cams_io'].values.astype(np.float32),
        target.values.astype(np.float32),
        patch_size=7,
        seq_len=3
    )
    
    # ==== Step 8: 保存处理后的数据 ====
    print("Step 8: 保存处理后的数据...")
    # 使用float32保存，减少内存和存储空间
    np.save(os.path.join(output_dir, "X_cnn.npy"), X_cnn.astype(np.float32))
    np.save(os.path.join(output_dir, "X_lstm.npy"), X_lstm.astype(np.float32))
    np.save(os.path.join(output_dir, "y.npy"), y.astype(np.float32))
    
    # 保存标准化参数
    np.save(os.path.join(output_dir, "norm_params.npy"), norm_params)
    # 保存归一化参数
    np.save(os.path.join(output_dir, "minmax_params.npy"), minmax_params)
    
    print(f"数据处理完成！数据已保存到: {output_dir}")

if __name__ == "__main__":
    process_and_save_data() 