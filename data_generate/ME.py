import MinkowskiEngine as ME
import torch

# 创建稀疏张量（仅包含一个有效点）
# coordinates = torch.tensor([[0, 0, 0, 0] ,[0, 1, 0, 0]])  # (batch_idx, x, y, z)
# features = torch.tensor([[1.0],[1.0]])  # 特征值为1
coordinates = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 1]])  # (batch_idx, x, y, z)
features = torch.tensor([[1.0],[1.0]])  # 特征值为1
sparse_tensor = ME.SparseTensor(
    features=features,
    coordinates=coordinates,
    tensor_stride=1  # 最底层分辨率
)

# 定义目标插值点（包括一个孤立点）
query_coords = torch.tensor([
    [0, 0, 0, 0],  # 与有效点重合
    [0, 1, 1, 1],   # 孤立点（无邻居）
    [0, .5, .5, .5],
    [0, 0, 0, 0.5],
]).float()

# 三线性插值
interp = ME.MinkowskiInterpolation()
output_features = interp(sparse_tensor, query_coords)

print(output_features)