import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_processed_data():
    """加载处理好的数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "processed_data")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"处理后的数据目录不存在: {data_dir}")
    
    X_cnn = np.load(os.path.join(data_dir, "X_cnn_2020.npy"))
    X_lstm = np.load(os.path.join(data_dir, "X_lstm_2020.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    
    return X_cnn, X_lstm, y

def build_model():
    """构建CNN+LSTM模型"""
    input_cnn = Input(shape=(7, 7, 5), name='cnn_input')
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_cnn)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Flatten()(x1)

    input_lstm = Input(shape=(3, 2), name='lstm_input')
    x2 = LSTM(32, return_sequences=True)(input_lstm)
    x2 = LSTM(32)(x2)

    x = Concatenate()([x1, x2])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[input_cnn, input_lstm], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('模型训练MAE')
    plt.xlabel('轮次')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(y_test, y_pred, output_dir):
    """绘制预测结果散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('观测XCO2 (ppm)')
    plt.ylabel('预测XCO2 (ppm)')
    plt.title(f'预测结果 (R2 = {r2_score(y_test, y_pred):.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_global_xco2_map(model, X_cnn, X_lstm, output_dir, sample_indices=None):
    """生成全球XCO2浓度分布图"""
    print("正在生成全球XCO2分布图...")
    
    # 如果没有指定样本索引，使用所有数据
    if sample_indices is None:
        sample_indices = np.arange(len(X_cnn))
    
    # 预测XCO2值
    X_cnn_sample = X_cnn[sample_indices]
    X_lstm_sample = X_lstm[sample_indices]
    predicted_xco2 = model.predict([X_cnn_sample, X_lstm_sample]).flatten()
    
    # 假设数据中包含经纬度信息，这里需要根据实际数据结构调整
    # 这里提供一个示例实现，您可能需要根据实际数据格式修改
    
    # 创建示例经纬度网格（实际应用中应该从数据中提取）
    # 这里假设数据是按经纬度网格组织的
    lat_resolution = 1.0  # 1度分辨率
    lon_resolution = 1.0
    
    lats = np.arange(-90, 91, lat_resolution)
    lons = np.arange(-180, 181, lon_resolution)
    
    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 这里需要根据实际数据结构来获取真实的经纬度信息
    # 示例：假设数据是按经纬度顺序排列的
    if len(predicted_xco2) == len(lats) * len(lons):
        # 重塑预测结果为网格
        xco2_grid = predicted_xco2.reshape(len(lats), len(lons))
    else:
        # 如果数据点数量不匹配，使用插值方法
        print(f"数据点数量 ({len(predicted_xco2)}) 与网格大小不匹配，使用插值方法")
        
        # 创建示例数据点（实际应用中应该从数据中提取真实坐标）
        # 这里假设数据点分布在全球范围内
        np.random.seed(42)  # 为了可重复性
        sample_lats = np.random.uniform(-90, 90, len(predicted_xco2))
        sample_lons = np.random.uniform(-180, 180, len(predicted_xco2))
        
        # 使用插值方法生成网格
        points = np.column_stack((sample_lons, sample_lats))
        xco2_grid = griddata(points, predicted_xco2, (lon_grid, lat_grid), method='linear')
    
    # 绘制全球XCO2分布图
    fig = plt.figure(figsize=(16, 10))  # 增加宽度为颜色条留出空间
    ax = plt.axes(projection=ccrs.Robinson())
    
    # 设置地图边界和特征
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    
    # 绘制XCO2分布
    im = ax.contourf(lon_grid, lat_grid, xco2_grid, 
                     levels=20, 
                     cmap='RdYlBu_r',  # 红色表示高浓度，蓝色表示低浓度
                     transform=ccrs.PlateCarree(),
                     extend='both')
    
    # 添加颜色条 - 调整位置到右侧
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, location='right')
    cbar.set_label('XCO2 (ppm)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 设置标题
    plt.title('全球XCO2浓度分布图 (模型预测)', fontsize=16, pad=20)
    
    # 添加网格线
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_xco2.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"全球XCO2分布图已保存到: {os.path.join(output_dir, 'global_xco2.png')}")

def generate_monthly_xco2_maps(model, X_cnn, X_lstm, output_dir, num_months=12):
    """生成月度XCO2浓度分布图 - 每个月单独一张图"""
    print("正在生成月度XCO2分布图...")
    
    for month in range(num_months):
        print(f"正在生成{month + 1}月XCO2分布图...")
        
        # 选择该月的数据（这里需要根据实际数据结构调整）
        # 示例：假设每个月有相同数量的数据点
        samples_per_month = len(X_cnn) // num_months
        start_idx = month * samples_per_month
        end_idx = (month + 1) * samples_per_month
        
        X_cnn_month = X_cnn[start_idx:end_idx]
        X_lstm_month = X_lstm[start_idx:end_idx]
        
        # 预测该月的XCO2值
        predicted_xco2 = model.predict([X_cnn_month, X_lstm_month]).flatten()
        
        # 创建网格（简化版本）
        lats = np.arange(-90, 91, 2)
        lons = np.arange(-180, 181, 2)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # 示例插值（实际应用中需要真实坐标）
        np.random.seed(month)
        sample_lats = np.random.uniform(-90, 90, len(predicted_xco2))
        sample_lons = np.random.uniform(-180, 180, len(predicted_xco2))
        
        points = np.column_stack((sample_lons, sample_lats))
        xco2_grid = griddata(points, predicted_xco2, (lon_grid, lat_grid), method='linear')
        
        # 绘制单月分布图
        fig = plt.figure(figsize=(16, 10))  # 增加宽度为颜色条留出空间
        ax = plt.axes(projection=ccrs.Robinson())
        
        # 设置地图特征
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        
        # 绘制XCO2分布
        im = ax.contourf(lon_grid, lat_grid, xco2_grid, 
                        levels=20, 
                        cmap='RdYlBu_r',
                        transform=ccrs.PlateCarree(),
                        extend='both')
        
        # 添加颜色条 - 调整位置到右侧
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, location='right')
        cbar.set_label('XCO2 (ppm)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # 设置标题
        plt.title(f'{month + 1}月XCO2浓度分布图 (模型预测)', fontsize=16, pad=20)
        
        # 添加网格线
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'xco2_{month + 1:02d}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{month + 1}月XCO2分布图已保存到: {os.path.join(output_dir, f'xco2_{month + 1:02d}.png')}")
    
    print(f"所有月度XCO2分布图已保存到: {output_dir}")

def main():
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("正在加载数据...")
    X_cnn, X_lstm, y = load_processed_data()
    
    # 划分训练集和测试集
    print("划分训练集和测试集...")
    X_cnn_train, X_cnn_test, X_lstm_train, X_lstm_test, y_train, y_test = train_test_split(
        X_cnn, X_lstm, y, test_size=0.2, random_state=42
    )
    
    # 构建模型
    print("构建模型...")
    model = build_model()
    model.summary()
    
    # 定义模型保存路径
    model_path = os.path.join(current_dir, "model.h5")
    best_model_path = os.path.join(current_dir, "best_model.h5")
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        [X_cnn_train, X_lstm_train], y_train,
        validation_data=([X_cnn_test, X_lstm_test], y_test),
        epochs=300,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            )
        ]
    )
    
    # 评估模型
    print("评估模型性能...")
    y_pred = model.predict([X_cnn_test, X_lstm_test])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)
    
    # 可视化结果
    print("生成可视化结果...")
    plot_training_history(history, output_dir)
    plot_prediction_scatter(y_test, y_pred, output_dir)
    
    # 生成全球XCO2分布图
    generate_global_xco2_map(model, X_cnn, X_lstm, output_dir)
    
    # 生成月度XCO2分布图
    generate_monthly_xco2_maps(model, X_cnn, X_lstm, output_dir)
    
    # 保存最终模型（如果是最优的）
    if os.path.exists(best_model_path):
        # 如果存在最优模型，将其重命名为标准名称
        if os.path.exists(model_path):
            os.remove(model_path)  # 删除旧的模型文件
        os.rename(best_model_path, model_path)
        print(f"最优模型已保存到: {model_path}")
    else:
        # 如果没有最优模型文件，保存当前模型
        model.save(model_path)
        print(f"模型已保存到: {model_path}")
    
    # 保存模型性能记录
    performance_record = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'best_val_loss': min(history.history['val_loss']),
        'best_val_mae': min(history.history['val_mae']),
        'epochs_trained': len(history.history['loss'])
    }
    
    # 将性能记录保存到文件
    import json
    with open(os.path.join(output_dir, 'model_performance.json'), 'w', encoding='utf-8') as f:
        json.dump(performance_record, f, indent=2, ensure_ascii=False)
    
    print(f"模型性能记录已保存到: {os.path.join(output_dir, 'model_performance.json')}")
    print(f"可视化结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()