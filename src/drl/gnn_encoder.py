# src/drl/gnn_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

class GNNEncoder(nn.Module):
    """
    使用图注意力网络 (GATv2) 对工作流状态图进行编码。
    它将一个图作为输入，输出一个代表全图的向量嵌入。
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        """
        初始化GNN编码器。
        Args:
            in_channels (int): 输入节点特征的维度。
            hidden_channels (int): 中间隐藏层的维度。
            out_channels (int): 输出图嵌入的维度。
            heads (int): GATv2中的多头注意力数量。
        """
        super(GNNEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            data (torch_geometric.data.Data): PyG格式的图数据对象，
                                              包含 x (节点特征), edge_index (边连接), batch (批次信息)。
        Returns:
            torch.Tensor: 图级别的嵌入向量。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GATv2卷积层
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # 全局平均池化，将节点嵌入聚合为图嵌入
        graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding