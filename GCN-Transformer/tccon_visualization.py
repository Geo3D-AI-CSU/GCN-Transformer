import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import pandas as pd
import scipy.stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_tccon_data():
    """加载所有TCCON站点数据"""
    tccon_files = sorted(glob.glob(os.path.join("ELYSIA", "tccon", "*.nc")))
    print(f"找到 {len(tccon_files)} 个TCCON站点数据文件")
    
    # 存储所有站点的信息
    stations_data = []
    
    for file in tccon_files:
        try:
            # 从文件名中提取站点代码
            station_code = os.path.basename(file).split('_')[0]
            
            # 读取数据
            ds = xr.open_dataset(file)
            
            # 打印数据集信息
            print(f"\n处理站点 {station_code}:")
            print("变量:", list(ds.variables.keys()))
            print("坐标:", list(ds.coords.keys()))
            
            # 提取站点信息
            station_info = {
                'station_code': station_code,
                'time_range': (ds.time.min().values, ds.time.max().values),
                'data_points': len(ds.time)
            }
            
            # 尝试获取经纬度信息
            if 'lat' in ds.variables and 'long' in ds.variables:
                lat = ds['lat'].values
                lon = ds['long'].values
                # 用众数（mode）
                latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
                longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
                station_info['latitude'] = latitude
                station_info['longitude'] = longitude
            elif 'latitude' in ds.variables and 'longitude' in ds.variables:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
                longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
                station_info['latitude'] = latitude
                station_info['longitude'] = longitude
            else:
                print(f"警告: 站点 {station_code} 没有找到经纬度信息")
                continue
            
            stations_data.append(station_info)
            ds.close()
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    return stations_data

def plot_stations_map(stations_data):
    """绘制站点分布图"""
    plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # 绘制站点
    for station in stations_data:
        if 'latitude' in station and 'longitude' in station:
            ax.scatter(station['longitude'], station['latitude'],
                      transform=ccrs.PlateCarree(),
                      s=50, alpha=0.7,
                      label=station['station_code'])
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title('TCCON站点全球分布')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tccon_stations_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_station_timeline(stations_data):
    """绘制站点时间线图"""
    plt.figure(figsize=(15, 8))
    
    # 按纬度排序站点
    sorted_stations = sorted(stations_data, key=lambda x: x.get('latitude', 0))
    
    for station in sorted_stations:
        start_date = pd.to_datetime(station['time_range'][0])
        end_date = pd.to_datetime(station['time_range'][1])
        
        plt.plot([start_date, end_date], 
                [station['station_code']] * 2,
                'o-', linewidth=2, markersize=8)
    
    plt.title('TCCON站点观测时间线')
    plt.xlabel('时间')
    plt.ylabel('站点代码')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tccon_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_points_distribution(stations_data):
    """绘制数据点分布图"""
    plt.figure(figsize=(15, 8))
    
    # 按数据点数量排序
    sorted_stations = sorted(stations_data, key=lambda x: x['data_points'])
    
    station_codes = [s['station_code'] for s in sorted_stations]
    data_points = [s['data_points'] for s in sorted_stations]
    
    plt.bar(station_codes, data_points)
    plt.title('TCCON各站点数据点数量分布')
    plt.xlabel('站点代码')
    plt.ylabel('数据点数量')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tccon_data_points.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_global_xco2_map():
    tccon_files = sorted(glob.glob(os.path.join("ELYSIA", "tccon", "*.nc")))
    lats, lons, xco2_means, station_codes = [], [], [], []
    for file in tccon_files:
        try:
            ds = xr.open_dataset(file)
            station_code = os.path.basename(file).split('_')[0]
            # 经纬度
            if 'lat' in ds.variables and 'long' in ds.variables:
                lat = ds['lat'].values
                lon = ds['long'].values
                latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
                longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
            elif 'latitude' in ds.variables and 'longitude' in ds.variables:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
                longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
            else:
                continue
            # xco2
            if 'xco2' in ds.variables:
                xco2 = ds['xco2'].values
                xco2_mean = float(np.nanmean(xco2))
            else:
                continue
            lats.append(latitude)
            lons.append(longitude)
            xco2_means.append(xco2_mean)
            station_codes.append(station_code)
            ds.close()
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    # 画图
    plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    sc = ax.scatter(lons, lats, c=xco2_means, cmap='viridis', s=120, edgecolor='k', vmin=390, vmax=420, zorder=10)
    for i, code in enumerate(station_codes):
        ax.text(lons[i], lats[i], code, fontsize=9, ha='center', va='bottom', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('TCCON站点XCO2均值 (ppm)')
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    plt.title('全球TCCON站点XCO2均值分布')
    plt.tight_layout()
    plt.savefig('tccon_global_xco2_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('全球TCCON站点XCO2均值分布图已保存为 tccon_global_xco2_map.png')

def plot_single_station_map(ds, station_code, output_dir='tccon_visualization/station_maps'):
    """为单个站点创建地图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取经纬度
    if 'lat' in ds.variables and 'long' in ds.variables:
        lat = ds['lat'].values
        lon = ds['long'].values
        latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
        longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
    else:
        print(f"警告: 站点 {station_code} 没有找到经纬度信息")
        return
    
    # 创建地图
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 设置地图范围（以站点为中心，显示周围5度范围）
    ax.set_extent([longitude-5, longitude+5, latitude-5, latitude+5], crs=ccrs.PlateCarree())
    
    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # 绘制站点位置
    ax.scatter(longitude, latitude, 
              transform=ccrs.PlateCarree(),
              s=100, color='red', edgecolor='black',
              label=station_code)
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # 添加标题和图例
    plt.title(f'TCCON站点 {station_code} 位置图')
    plt.legend(loc='upper right')
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{station_code}_map.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存站点 {station_code} 的地图到: {output_file}")

def plot_single_station_xco2(ds, station_code, output_dir='tccon_visualization/station_xco2'):
    """为单个站点创建XCO2时间序列图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有XCO2数据
    if 'xco2' not in ds.variables:
        print(f"警告: 站点 {station_code} 没有XCO2数据")
        return
    
    # 获取站点位置信息
    location = ds.attrs.get('short_location', '')
    if not location:
        location = ds.attrs.get('location', '')
    
    # 获取经纬度信息
    if 'lat' in ds.variables and 'long' in ds.variables:
        lat = ds['lat'].values
        lon = ds['long'].values
        latitude = float(scipy.stats.mode(lat, keepdims=True).mode[0])
        longitude = float(scipy.stats.mode(lon, keepdims=True).mode[0])
        location_info = f"{location} ({latitude:.2f}°N, {longitude:.2f}°E)"
    else:
        location_info = location
    
    # 获取时间和XCO2数据
    time = pd.to_datetime(ds.time.values)
    xco2 = ds.xco2.values
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制XCO2时间序列
    plt.plot(time, xco2, 'b.', alpha=0.5, markersize=2)
    
    # 添加趋势线
    z = np.polyfit(range(len(time)), xco2, 1)
    p = np.poly1d(z)
    plt.plot(time, p(range(len(time))), "r--", linewidth=2)
    
    # 设置图形属性
    plt.title(f'TCCON站点 {station_code} - {location_info}\nXCO2时间序列', fontsize=12)
    plt.xlabel('时间', fontsize=10)
    plt.ylabel('XCO2 (ppm)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_xco2 = np.nanmean(xco2)
    std_xco2 = np.nanstd(xco2)
    plt.text(0.02, 0.98, 
             f'平均值: {mean_xco2:.2f} ppm\n标准差: {std_xco2:.2f} ppm',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{station_code}_xco2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存站点 {station_code} ({location_info}) 的XCO2时间序列图到: {output_file}")

def main():
    # 创建输出目录
    os.makedirs('tccon_visualization', exist_ok=True)
    
    # 获取所有TCCON文件
    tccon_files = sorted(glob.glob(os.path.join("ELYSIA", "tccon", "*.nc")))
    print(f"找到 {len(tccon_files)} 个TCCON站点数据文件")
    
    # 为每个站点创建单独的地图
    for file in tccon_files:
        try:
            # 从文件名中提取站点代码
            station_code = os.path.basename(file).split('_')[0]
            print(f"\n处理站点 {station_code}")
            
            # 读取数据
            ds = xr.open_dataset(file)
            
            # 创建该站点的地图
            plot_single_station_map(ds, station_code)
            
            # 创建该站点的XCO2时间序列图
            plot_single_station_xco2(ds, station_code)
            
            ds.close()
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    print("\n所有站点地图和XCO2时间序列图创建完成！")

if __name__ == "__main__":
    main()
    plot_global_xco2_map() 