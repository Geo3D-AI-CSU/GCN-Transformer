# -*- coding: utf-8 -*-
"""
GCN + Transformer 训练与评估脚本
"""

# -------------------- 导入依赖模块 --------------------
import os  # 操作系统交互（路径、文件夹等）
import numpy as np  # 数值计算库
import torch  # PyTorch 张量与模型库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # 常用的函数式接口
from torch_geometric.nn import GCNConv, global_mean_pool  # PyG 的 GCN 层与池化
from torch_geometric.data import Data, DataLoader  # PyG 的数据容器与加载器
from sklearn.model_selection import train_test_split  # 用于数据切分
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 回归评价指标
import matplotlib.pyplot as plt  # 绘图
import seaborn as sns  # 辅助绘图美化
import cartopy.crs as ccrs  # 地图投影
import cartopy.feature as cfeature  # 地图特征
from scipy.interpolate import griddata  # 插值工具
import pandas as pd  # 表格与时间处理
from torch_geometric.data import InMemoryDataset  # PyG 的内存数据集基类
import json  # 读写 json
from tqdm import tqdm  # 进度条工具
import argparse  # 命令行解析
from scipy import stats  # 统计工具

# 设置绘图中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为 SimHei，使图表可以显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 使负号能够正常显示

# 设置训练设备
device = torch.device('gpu')
print(f"使用设备: {device}")  # 打印所用设备


# -------------------- 模型定义 --------------------
class GCNTransformerModel(nn.Module):
    """融合 GCN 与 Transformer 的回归模型（逐时间步处理图数据，最后输出每个节点的回归值）。
    主要思路：对每个时间步 t，使用 GCN 层处理图上的特征；对时序维使用 Transformer 编码器进行建模；最后输出每个节点的值。
    参数说明见 __init__ 的注释。
    """
    def __init__(self, num_features=5, hidden_dim=64, num_layers=3, num_heads=8, dropout=0.1,
                 use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                 pooling='last_time'):
        super(GCNTransformerModel, self).__init__()
        # 保存配置项到实例变量，便于后续分支逻辑判断
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_gcn = use_gcn
        self.use_transformer = use_transformer
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.use_clamp = use_clamp
        self.pooling = pooling

        # ---------- 定义 GCN 层列表 ----------
        self.gcn_layers = nn.ModuleList()
        # 第一层将输入维度 num_features 映射到 hidden_dim
        self.gcn_layers.append(GCNConv(num_features, hidden_dim))
        # 后续层为 hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # 如果不使用 GCN，可以通过线性层将原始特征投影到 hidden_dim
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # ---------- 定义 Transformer 编码器（可选） ----------
        if self.use_transformer:
            # 使用 PyTorch 原生的 TransformerEncoderLayer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,  # Transformer 的特征维度需与 GCN 输出一致
                nhead=num_heads,  # 多头注意力头数
                dim_feedforward=hidden_dim * 4,  # 前馈网络维度（通常为 4*d_model）
                dropout=dropout,
                batch_first=True  # 使用 (batch, seq, feature) 的输入顺序
            )
            # 堆叠两层编码器（可配置）
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.transformer = None

        # ---------- 输出层：把 transformer 或池化后的向量映射为单值回归输出 ----------
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ---------- 批归一化层（可选） ----------
        if self.use_batch_norm:
            # 为每一层 GCN 准备一个 BatchNorm1d（作用于节点维度）
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.batch_norms = nn.ModuleList()

    def forward(self, x, edge_index, batch=None):
        """前向传播：
        x: 张量，可能形状为 [seq_len, num_nodes, num_features]（完整时序）
           或 [num_nodes, num_features]（单帧）或 [batch*num_nodes, num_features]（分批展开）
        edge_index: 图的边索引，形状 [2, E]
        batch: 若为批处理，PyG 的 batch 张量，用于指示节点到样本的映射
        返回：每个节点的预测值（一维向量）
        """
        # 处理不同输入形态，统一为 [seq_len, batch_size*num_nodes, num_features]
        if len(x.shape) == 3:
            # 若 x 已经是时序输入，解包出 seq_len、num_nodes、num_features
            seq_len, num_nodes, num_features = x.shape
            batch_size = 1  # 若没有 batch 维，则视为单样本批
        else:
            # 若 x 不是 3D，可能是 2D 批或单帧
            seq_len = 1
            if batch is not None:
                # 当提供 batch 时，推断批次大小
                batch_size = batch.max().item() + 1
                num_nodes = x.shape[0] // batch_size
            else:
                batch_size = 1
                num_nodes = x.shape[0]
            x = x.unsqueeze(0)  # 扩展为 3D： [1, num_nodes, num_features]

        # 如果第一个维度确实等于 seq_len，重新组织形状
        if len(x.shape) == 3 and x.shape[0] == seq_len:
            x = x.view(seq_len, batch_size * num_nodes, num_features)

        # ---------- 对每个时间步 t 应用 GCN ----------
        gcn_outputs = []  # 收集每个时间步的 GCN 输出
        for t in range(seq_len):
            x_t = x[t]  # 当前时间帧，形状 [batch_size*num_nodes, num_features]
            if self.use_gcn:
                # 对当前时间帧依次使用 GCN 层
                for i, gcn_layer in enumerate(self.gcn_layers):
                    # 修正 edge_index（如果 edge_index 的索引大于当前节点数则映射回节点数范围）
                    if edge_index.max() >= x_t.shape[0]:
                        edge_index_mapped = edge_index.clone() % x_t.shape[0]
                    else:
                        edge_index_mapped = edge_index
                    # GCN 卷积
                    x_t = gcn_layer(x_t, edge_index_mapped)
                    # 可选：clamp 防止数值爆炸
                    if self.use_clamp:
                        x_t = torch.clamp(x_t, -100, 100)
                    # 批归一化（若定义）
                    if i < len(self.batch_norms):
                        x_t = self.batch_norms[i](x_t)
                    # 激活
                    x_t = F.relu(x_t)
                    # dropout
                    if self.use_dropout:
                        x_t = F.dropout(x_t, p=0.1, training=self.training)
            else:
                # 不使用 GCN 时，直接通过线性投影获得隐藏表示
                x_t = self.input_proj(x_t)
                x_t = F.relu(x_t)
                if self.use_dropout:
                    x_t = F.dropout(x_t, p=0.1, training=self.training)
            # 将当前时间步的输出保存
            gcn_outputs.append(x_t)

        # 把时间步输出堆叠： shape -> [seq_len, batch_size*num_nodes, hidden_dim]
        gcn_output = torch.stack(gcn_outputs, dim=0)

        # 重新组织为 [seq_len, batch_size, num_nodes, hidden_dim]
        gcn_output = gcn_output.view(seq_len, batch_size, num_nodes, self.hidden_dim)
        # 置换为 [batch_size, seq_len, num_nodes, hidden_dim]
        gcn_output = gcn_output.permute(1, 0, 2, 3)
        # 把 batch 与节点合并，形成 Transformer 可接受的序列维：[batch_size * num_nodes, seq_len, hidden_dim]
        gcn_output = gcn_output.view(batch_size * num_nodes, seq_len, self.hidden_dim)
        if self.use_clamp:
            gcn_output = torch.clamp(gcn_output, -100, 100)

        # ---------- 使用 Transformer 处理时序（可选） ----------
        if self.transformer is not None:
            transformer_output = self.transformer(gcn_output)  # 输出形状 [batch_size*num_nodes, seq_len, hidden_dim]
            if self.use_clamp:
                transformer_output = torch.clamp(transformer_output, -100, 100)
            # 取序列最后一时刻的向量作为节点表示
            final_output = transformer_output[:, -1, :]
        else:
            # 若不使用 Transformer，可采用不同的时序池化策略
            if self.pooling == 'mean_time':
                final_output = gcn_output.mean(dim=1)  # 平均时序
            else:
                final_output = gcn_output[:, -1, :]  # 取最后时刻

        # ---------- 输出层映射为单值 ----------
        output = self.output_layer(final_output)  # [batch_size*num_nodes, 1]
        if self.use_clamp:
            output = torch.clamp(output, -100, 100)
        output = output.squeeze(-1)  # [batch_size*num_nodes]

        # ---------- 处理输出大小与期望不一致的情况 ----------
        expected_output_size = x.shape[1] if len(x.shape) == 3 else x.shape[0]
        if output.shape[0] != expected_output_size:
            # 若输出比期望长则截断，比期望短则用 0 填充
            if output.shape[0] > expected_output_size:
                output = output[:expected_output_size]
            else:
                padding = torch.zeros(expected_output_size - output.shape[0], device=output.device)
                output = torch.cat([output, padding])
        return output


# -------------------- 数据加载函数 --------------------
def load_processed_data():
    """加载事先处理好的 GCN+Transformer 样本文件（gcn_transformer_data.pt）。
    返回：samples（list 或者 PyG 的 Data 对象列表）
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
    data_dir = os.path.join(current_dir, "processed_data")  # 预期的处理后数据目录

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"处理后的数据目录不存在: {data_dir}")

    # 样本文件路径
    samples_path = os.path.join(data_dir, "gcn_transformer_data.pt")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"GCN+Transformer样本文件不存在: {samples_path}")

    print("正在加载GCN+Transformer样本...")
    # 使用 torch.load 读取序列化的样本列表；weights_only=False 允许加载完整对象
    samples = torch.load(samples_path, weights_only=False)

    return samples


# -------------------- 自定义损失（带 mask） --------------------
def create_masked_loss():
    """返回一个带缺失值掩码的 MSE 损失函数，使模型训练忽略标签中的 NaN。"""
    def masked_mse_loss(pred, target, mask=None):
        # 把 pred/target 中的 NaN/Inf 转为有限值，避免运算炸掉
        pred = torch.nan_to_num(pred, nan=0.0, posinf=100, neginf=-100)
        target = torch.nan_to_num(target, nan=0.0, posinf=100, neginf=-100)

        # 如果 target 是二维（batch_size, seq_len），取最后时间步作为目标
        if target.dim() == 2 and target.shape[1] > 1:
            target = target[:, -1]

        # 展平为 1D 向量以便掩码选择
        pred = pred.view(-1)
        target = target.view(-1)

        # 若长度不匹配，截取到最短长度
        min_len = min(pred.shape[0], target.shape[0])
        pred = pred[:min_len]
        target = target[:min_len]

        if mask is None:
            # 默认掩码为 target 非 NaN
            mask = ~torch.isnan(target)

        # 保证 mask 长度一致
        mask = mask[:min_len]

        valid_pred = pred[mask]
        valid_target = target[mask]

        # 限幅，防止异常值造成梯度问题
        valid_pred = torch.clamp(valid_pred, -100, 100)
        valid_target = torch.clamp(valid_target, -100, 100)

        if len(valid_pred) == 0:
            # 若没有有效样本，返回一个可求导的 0 张量
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(valid_pred, valid_target)
    return masked_mse_loss


# -------------------- 训练过程 --------------------
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001):
    """训练循环，包含早停和学习率调度。返回训练损失与验证损失曲线。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 当验证集指标不再下降时缩小学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
    criterion = create_masked_loss()  # 使用定制损失

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # 早停耐心

    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", position=0)
    for epoch in epoch_pbar:
        model.train()
        train_loss, num_batches = 0.0, 0
        # 训练阶段：遍历训练集的每个批次
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", leave=False)):
            # 跳过包含 NaN/Inf 的批次
            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print(f"跳过批次 {batch_idx}：x 含 NaN/Inf")
                continue
            if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                print(f"跳过批次 {batch_idx}：y 含 NaN/Inf")
                continue
            optimizer.zero_grad()
            # 前向传播：将批次移动到 device
            pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
            loss = criterion(pred, batch.y.to(device))
            # 若损失无效则跳过
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"跳过批次 {batch_idx}：loss 无效")
                continue
            loss.backward()
            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / max(1, num_batches)

        # 验证阶段
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"验证 Epoch {epoch+1}", leave=False)):
                pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
                loss = criterion(pred, batch.y.to(device))
                val_loss += loss.item()
                val_batches += 1
        val_loss = val_loss / max(1, val_batches)

        # 记录损失并调整学习率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # 早停逻辑：若验证损失下降则保存模型并重置计数器
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gcn_transformer_model.pth')
        else:
            patience_counter += 1
        epoch_pbar.set_postfix({"训练损失": train_loss, "验证损失": val_loss})
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break
    return train_losses, val_losses


# -------------------- 模型评估 --------------------
def evaluate_model(model, test_loader):
    """在测试集上进行评估，返回 RMSE、MAE、R2 以及预测与真值数组。"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估进度")):
            # 跳过含 NaN/Inf 的批次
            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print(f"跳过批次 {batch_idx}：x 含 NaN/Inf")
                continue
            if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                print(f"跳过批次 {batch_idx}：y 含 NaN/Inf")
                continue
            pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))

            # 处理目标 y 的形状：若为二维且有多个时序步，取最后一步
            if batch.y.dim() == 2 and batch.y.shape[1] > 1:
                target_y = batch.y[:, -1]
            else:
                target_y = batch.y

            # 确保长度一致
            min_len = min(pred.shape[0], target_y.shape[0])
            pred = pred[:min_len]
            target_y = target_y[:min_len]

            mask = ~torch.isnan(target_y)
            mask = mask[:min_len]  # 确保 mask 长度一致

            all_preds.append(pred[mask].cpu().numpy())
            all_targets.append(target_y[mask].cpu().numpy())
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
    else:
        rmse, mae, r2 = float('inf'), float('inf'), 0.0
    return rmse, mae, r2, all_preds, all_targets


# -------------------- 可视化工具 --------------------
def plot_pred_vs_tccon_density(y_true, y_pred, output_dir):
    """绘制高质量的密度散点图，并保存为 PNG/PDF。"""
    print("正在绘制高质量密度散点图...")
    # 统计评价指标
    r2 = r2_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else float('inf')
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else float('inf')
    bias = float(np.mean(y_pred - y_true)) if len(y_true) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 10))

    # 计算点密度（用于颜色映射）
    print("计算点密度用于着色...")
    xy = np.vstack([y_true, y_pred])
    try:
        z = stats.gaussian_kde(xy)(xy)  # KDE 估计点密度
    except Exception:
        z = np.ones_like(y_true)  # 若失败，则退化为均匀密度

    # 绘制散点图，颜色根据密度 z
    scatter = ax.scatter(y_true, y_pred, c=z, cmap='viridis', alpha=0.6, s=12, edgecolors='white', linewidth=0.2)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Point Density', fontsize=12, fontweight='bold')

    # 绘制 1:1 参考线
    plot_min = min(np.min(y_true), np.min(y_pred))
    plot_max = max(np.max(y_true), np.max(y_pred))
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, alpha=0.8, label='1:1 Line')

    # 坐标与网格设置
    ax.set_xlim(plot_min - 1, plot_max + 1)
    ax.set_ylim(plot_min - 1, plot_max + 1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 标签与标题
    ax.set_xlabel('TCCON XCO2 [ppm]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted XCO2 [ppm]', fontsize=14, fontweight='bold')
    ax.set_title('Predicted vs TCCON XCO2', fontsize=16, fontweight='bold', pad=20)

    # 在图上添加统计框
    stats_text = f"""Statistics:\nR² = {r2:.4f}\nRMSE = {rmse:.2f} ppm\nMAE = {mae:.2f} ppm\nBias = {bias:.2f} ppm"""
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # 根据 R² 生成文件名标签并保存
    r2_tag = f"{int(round(r2*100)):03d}"
    png_path = os.path.join(output_dir, f"CAMS_vs_TCCON_R2_{r2_tag}.png")
    pdf_path = os.path.join(output_dir, f"CAMS_vs_TCCON_R2_{r2_tag}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存高质量散点图: {png_path}")


def plot_training_history(train_losses, val_losses, output_dir):
    """绘制并保存训练/验证损失历史曲线（对数与线性尺度）。"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('模型训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.yscale('log')  # 左图使用对数尺度显示损失

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('模型训练损失 (线性尺度)')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcn_transformer_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_scatter(y_test, y_pred, output_dir):
    """绘制测试集上的真实值 vs 预测值 散点图并保存。"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('观测XCO2 (ppm)')
    plt.ylabel('预测XCO2 (ppm)')
    plt.title(f'GCN+Transformer预测结果 (R2 = {r2_score(y_test, y_pred):.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcn_transformer_prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()


# -------------------- 全球图生成 --------------------
def generate_global_xco2_map(model, samples, output_dir, sample_indices=None):
    """使用模型预测并绘制全球 XCO2 分布图（简单栅格化假设网格为正方形）。
    注意：此函数假设模型输出可以整齐重塑为正方形网格（grid_size x grid_size），实际需与数据预处理一致。
    """
    print("正在生成全球XCO2分布图...")
    model.eval()

    # 如果没有指定索引，默认仅绘制第一个样本
    if sample_indices is None:
        sample_indices = [0]

    with torch.no_grad():
        for idx in sample_indices:
            sample = samples[idx].to(device)
            pred = model(sample.x, sample.edge_index)

            # 计算网格大小：假设节点数为 grid_size^2
            grid_size = int(np.sqrt(len(pred)))
            pred_grid = pred.cpu().numpy().reshape(grid_size, grid_size)

            # 创建经纬度网格（这里简单使用等间距，经纬范围 -90..90、-180..180）
            lats = np.linspace(-90, 90, grid_size)
            lons = np.linspace(-180, 180, grid_size)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # 绘图（Robinson 投影）
            fig = plt.figure(figsize=(16, 10))
            ax = plt.axes(projection=ccrs.Robinson())
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)

            im = ax.contourf(lon_grid, lat_grid, pred_grid,
                           levels=20,
                           cmap='RdYlBu_r',
                           transform=ccrs.PlateCarree(),
                           extend='both')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, location='right')
            cbar.set_label('XCO2 (ppm)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            plt.title(f'全球XCO2浓度分布图 (GCN+Transformer预测) - 样本{idx}', fontsize=16, pad=20)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'gcn_transformer_global_xco2_sample_{idx}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    print(f"全球XCO2分布图已保存到: {output_dir}")


# -------------------- 消融实验套件 --------------------
def run_ablation_suite(processed_samples, output_dir, epochs=10, batch_size=2):
    """运行一系列消融实验：切换是否使用 transformer/gcn/batchnorm/dropout 等，并记录结果。"""
    os.makedirs(output_dir, exist_ok=True)

    def build_loaders(local_samples, bs):
        """按时间顺序划分训练/验证/测试并返回 DataLoader。"""
        total = len(local_samples)
        train_end = int(total * 0.72)
        val_end = int(total * 0.80)
        train_samples = local_samples[:train_end]
        val_samples = local_samples[train_end:val_end]
        test_samples = local_samples[val_end:]
        train_loader = DataLoader(train_samples, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_samples, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_samples, batch_size=bs, shuffle=False)
        return train_loader, val_loader, test_loader

    def slice_features(samples, feature_indices):
        """如果给出 feature_indices，则对每个样本的 x 按通道切片（用于只保留部分特征的消融）。"""
        if feature_indices is None:
            return samples
        sliced = []
        for s in samples:
            x = s.x
            if x.ndim == 3:
                new_x = x[:, :, feature_indices]
            elif x.ndim == 2:
                new_x = x[:, feature_indices]
            else:
                raise ValueError(f"x 的维度异常: {x.shape}")
            new_data = Data(x=new_x, y=s.y, edge_index=s.edge_index)
            sliced.append(new_data)
        return sliced

    # 定义多个实验配置
    experiments = []
    # 基线
    experiments.append({
        'name': 'full_model',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 无Transformer
    experiments.append({
        'name': 'no_transformer',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=False, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 无GCN
    experiments.append({
        'name': 'no_gcn',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=False, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 无BatchNorm
    experiments.append({
        'name': 'no_batchnorm',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=False, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 无Dropout
    experiments.append({
        'name': 'no_dropout',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.0,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=False, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 无Clamp
    experiments.append({
        'name': 'no_clamp',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=False,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 更少GCN层
    experiments.append({
        'name': 'gcn_layers_1',
        'model_kwargs': dict(num_features=5, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': None
    })
    # 仅CAMS特征（举例）
    experiments.append({
        'name': 'features_cams_only',
        'model_kwargs': dict(num_features=2, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': [0, 1]
    })
    # 去掉CAMS-IO（举例）
    experiments.append({
        'name': 'features_wo_cams_io',
        'model_kwargs': dict(num_features=4, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                             use_gcn=True, use_transformer=True, use_batch_norm=True, use_dropout=True, use_clamp=True,
                             pooling='last_time'),
        'feature_indices': [0, 2, 3, 4]
    })

    results = []
    for exp in experiments:
        name = exp['name']
        feature_indices = exp['feature_indices']
        local_samples = slice_features(processed_samples, feature_indices)
        train_loader, val_loader, test_loader = build_loaders(local_samples, batch_size)
        model_kwargs = exp['model_kwargs']
        model = GCNTransformerModel(**model_kwargs)
        print(f"\n===== 运行实验: {name} =====")
        train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=epochs)
        rmse, mae, r2, y_pred, y_test = evaluate_model(model, test_loader)
        print(f"实验 {name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        results.append({
            'name': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'last_train_loss': train_losses[-1] if len(train_losses) > 0 else None,
            'last_val_loss': val_losses[-1] if len(val_losses) > 0 else None
        })

    # 写入 CSV 保存消融结果
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'rmse', 'mae', 'r2', 'last_train_loss', 'last_val_loss'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n✅ 消融实验完成，结果已保存到: {csv_path}")


# -------------------- 主流程 --------------------
def main(args):
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 加载处理后的样本
    print("正在加载数据...")
    samples = load_processed_data()

    print("检查数据完整性并修复边索引...")
    valid_samples = []

    # 统计原始数据状况
    total_samples = len(samples)
    nan_samples = 0
    inf_samples = 0
    valid_y_samples = 0

    for i, sample in enumerate(tqdm(samples, desc="检查样本", unit="个")):
        try:
            # 检查 y 的有效性
            y = sample.y
            has_nan = torch.isnan(y).any()
            has_inf = torch.isinf(y).any()
            has_valid_y = (~torch.isnan(y) & ~torch.isinf(y)).any()

            if has_nan:
                nan_samples += 1
            if has_inf:
                inf_samples += 1
            if has_valid_y:
                valid_y_samples += 1

            # 仅保留含有有效 y 值的样本
            if not has_valid_y:
                continue

            # 规范化 x 的形状为 [seq_len, num_nodes, num_features]
            if sample.x.ndim == 3:
                seq_len, num_nodes, num_features = sample.x.shape
            elif sample.x.ndim == 2:
                num_nodes, num_features = sample.x.shape
                seq_len = 1
                sample.x = sample.x.unsqueeze(0)  # 升为 3 维
            else:
                raise ValueError(f"x 的维度异常: {sample.x.shape}")

            # 检查并修复 edge_index（若索引超范围或负值）
            edge_index = sample.edge_index.cpu()
            if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
                unique_nodes, new_ids = torch.unique(edge_index, sorted=True, return_inverse=True)
                sample.edge_index = new_ids.view_as(edge_index)
            else:
                sample.edge_index = edge_index

            # 转为合适的数据类型（long/float）并移到 CPU 以便后续处理
            sample.edge_index = sample.edge_index.long()
            sample.x = sample.x.float().cpu()
            sample.y = sample.y.float().cpu()

            # 跳过包含 NaN 或 inf 的样本
            if torch.isnan(sample.x).any() or torch.isinf(sample.x).any():
                print(f"样本 {i} 的 x 包含 NaN 或 inf，跳过")
                continue

            # 将数值限制到合理范围，避免异常值
            sample.x = torch.clamp(sample.x, -100, 100)
            sample.y = torch.clamp(sample.y, -100, 100)

            valid_samples.append(sample)

        except Exception as e:
            print(f"❌ 样本 {i} 修复失败: {e}")
            continue

    # 数据统计信息输出
    print(f"\n=== 数据统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"包含 NaN 的样本: {nan_samples}")
    print(f"包含 inf 的样本: {inf_samples}")
    print(f"有有效 y 值的样本: {valid_y_samples}")
    print(f"最终有效样本: {len(valid_samples)}")
    print(f"数据利用率: {len(valid_samples)/total_samples*100:.1f}%")
    print("=" * 50)

    print(f"✅ 有效样本数量: {len(valid_samples)} / {len(samples)}")

    # 检查样本数量
    if len(valid_samples) == 0:
        raise ValueError("没有找到有效的训练样本！请检查数据预处理步骤。")

    if len(valid_samples) < 100:
        print(f"⚠️  警告: 有效样本数量较少 ({len(valid_samples)})，可能影响训练效果")

    samples = valid_samples

    # 若命令行指定运行消融套件，则运行并返回
    if getattr(args, 'ablation', False):
        run_ablation_suite(samples, output_dir, epochs=getattr(args, 'epochs', 10), batch_size=getattr(args, 'batch_size', 2))
        return

    # 按时间顺序划分训练/验证/测试集，防止未来信息泄露
    print("划分数据集...")
    total_samples = len(samples)
    train_end = int(total_samples * 0.72)  # 72% 用于训练
    val_end = int(total_samples * 0.80)    # 下一个 8% 用于验证，其余用于测试

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"测试集: {len(test_samples)} 样本")

    # 创建 DataLoader
    train_loader = DataLoader(train_samples, batch_size=getattr(args, 'batch_size', 2), shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=getattr(args, 'batch_size', 2), shuffle=False)
    test_loader = DataLoader(test_samples, batch_size=getattr(args, 'batch_size', 2), shuffle=False)

    # 构建模型（可调整超参数）
    print("构建GCN+Transformer模型...")
    model = GCNTransformerModel(
        num_features=5,  # CAMS-EGG4, CAMS-IO, wind_u, wind_v, wind_speed
        hidden_dim=16,   # 隐藏维度
        num_layers=2,    # GCN 层数
        num_heads=2,     # Transformer 头数
        dropout=0.2      # dropout 比例
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=getattr(args, 'epochs', 30))

    # 评估模型
    print("评估模型性能...")
    rmse, mae, r2, y_pred, y_test = evaluate_model(model, test_loader)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # 可视化结果并保存
    print("生成可视化结果...")
    plot_training_history(train_losses, val_losses, output_dir)
    plot_prediction_scatter(y_test, y_pred, output_dir)
    plot_pred_vs_tccon_density(y_test, y_pred, output_dir)
    generate_global_xco2_map(model, samples, output_dir, sample_indices=[0, 100, 500])
    print(f"可视化结果已保存到: {output_dir}")


# -------------------- 脚本入口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true', help='运行消融实验套件')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    args = parser.parse_args()
    main(args)
