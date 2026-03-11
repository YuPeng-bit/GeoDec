import json
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MVBenchmarkDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 split_json, 
                 seed=42, 
                 num_samples=100,
                 source_type='rand6', # 'rand6' or 'ortho_front'
                 transform=None):
        """
        Args:
            root_dir: 数据集根目录 (e.g., ./data)
            split_json: objaverse_rest.json 的路径
            seed: 随机种子
            num_samples: 采样数量
            source_type: 输入源类型。'rand6'使用随机视角的图片作为输入，'ortho_front'使用正视图(0000)作为输入。
        """
        self.root_dir = root_dir
        self.ortho_dir = os.path.join(root_dir, "texture_ortho10view_easylight_objaverse")
        self.rand_dir = os.path.join(root_dir, "texture_rand_easylight_objaverse")
        self.transform = transform
        
        # 1. 加载并筛选样本ID
        with open(split_json, 'r') as f:
            all_ids = json.load(f)
        
        # 2. 随机采样
        random.seed(seed)
        # 确保样本ID是字符串格式
        all_ids = [str(x) for x in all_ids]
        # 如果样本数足够，进行无放回采样
        self.sample_ids = random.sample(all_ids, min(num_samples, len(all_ids)))
        self.sample_ids.sort() # 排序保证多卡/多次运行顺序一致

        # 3. 定义 GT 的映射关系
        # 模型输出顺序 (假设): [Front, Front-Left, Left, Back, Right, Front-Right]
        # 对应方位角: [0, 45, 90, 180, 270, 315]
        # 对应文件编号: [0000, 0004, 0001, 0002, 0003, 0005]
        self.gt_indices = ["0000", "0004", "0001", "0002", "0003", "0005"]
        self.source_type = source_type

    def __len__(self):
        return len(self.sample_ids)

    def _get_path(self, base_dir, uid, filename):
        # 逻辑：目录结构为 base_dir/uid[:2]/uid/filename
        return os.path.join(base_dir, uid[:2], uid, filename)

    def __getitem__(self, idx):
        uid = self.sample_ids[idx]
        
        # =========== 核心修改开始 ===========
        # 论文评估标准：Input view between -45 and 45.
        # 你的 Rand6 数据集是 0-360 随机的，直接用会导致坐标系未对齐。
        # 解决方案：使用 Ortho10View 中的 '0000' (正视图, 0度) 作为输入。
        # 这符合论文定义的 range，且能保证坐标系绝对对齐。
        
        # 强制指定 source 为 ortho 的 0000 号图
        source_path = self._get_path(self.ortho_dir, uid, "color_0000.webp")
        # =========== 核心修改结束 ===========
            
        # --- 准备 Target Images (GT) ---
        gt_paths = []
        for index in self.gt_indices:
            gt_path = self._get_path(self.ortho_dir, uid, f"color_{index}.webp")
            gt_paths.append(gt_path)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file missing: {source_path}")
        
        return {
            "uid": uid,
            "source_path": source_path,
            "gt_paths": gt_paths
        }