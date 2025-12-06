import torch
from torch.utils.data import Dataset

from pathlib import Path
import json
import numpy as np
import sys


class RawPointCloudDataset(Dataset):
    def __init__(
        self,
        min_pointclouds: int = 10,
        max_pointclouds: int = 150,
        max_seq_length: int = 70,
        input_seq_length: int = 3,
        feats: int = 4,
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.max_pointclouds = max_pointclouds
        self.input_seq_length = input_seq_length
        self.feats = feats

        results = []

        root_dirs = ["data/data_an-20251129T074104Z-1-001/lungham1021"]
        for root in root_dirs:
            root = Path(f"{root}")
            for k, p in enumerate(root.rglob("*json")):

                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)["data"]
                    for i in range(0, len(data) - max_seq_length):

                        sys.stdout.write(f"\nprocessing {k} / {i}")
                        sys.stdout.flush()
                        frame_seq = data[i : i + max_seq_length]
                        pointcloud_seq = [
                            frame["frameData"]["pointCloud"]
                            for frame in frame_seq
                            if frame.get("frameData")
                        ]
                        pointcloud_seq = [
                            np.array(pc)[:, :self.feats]
                            for pc in pointcloud_seq
                            if len(pc) > 0
                        ]
                        if len(pointcloud_seq) < len(frame_seq):
                            continue
                            
                        all_points = np.concatenate(pointcloud_seq, axis=0)
                        median_xyz = np.median(all_points[:, :2], axis=0)
                        pointcloud_seq = [p[:, :-1] for p in pointcloud_seq]
                        tvoxel = self.voxelize(pointcloud_seq, median_xyz)


                        results.append({"chunks": tvoxel})
                        
        # import matplotlib.pyplot as plt
        # for i in range(len(results)):

        #     for j in range(30):
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111, projection='3d')
        #         temp = results[i]["chunks"][j]

        #         ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], s=10)

        #         ax.set_xlim(left=-3, right=3)
        #         ax.set_ylim(bottom=-3, top=3)
        #         ax.set_zlim(bottom=-3, top=3)
        #         ax.set_xlabel('X')
        #         ax.set_ylabel('Y')
        #         ax.set_zlabel('Z')
        #         ax.set_title(f"number of point clouds: {len(temp)}")
        #         plt.show()
        self.results = results

    def collater(self, pc):
        return {"x": torch.tensor(pc, dtype=torch.float32)}
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, index):
        data  = self.results[index]["chunks"]
        return data

    def voxelize(self, true_data2, cluster_center):
        # === 1. 定義空間範圍 ===
        first_3 = np.concatenate(true_data2[:3], axis=0)   # 第0~2幀
        mid_3   = np.concatenate(true_data2[2:5], axis=0)  # 第2~4幀
        last_3  = np.concatenate(true_data2[4:7], axis=0)  # 第4~6幀
        true_data2 = [first_3, mid_3, last_3]              # 更新為三幀

        cx, cy = cluster_center
        x_range = 2.0
        y_range = 2.0
        z_min, z_max = -0.2, 2.0
        x_min, x_max = cx - x_range / 2, cx + x_range / 2
        y_min, y_max = cy - y_range / 2, cy + y_range / 2
        x_size, y_size, z_size = 40, 40, 30

        # === 2. 建立離散 bin 中心 ===
        lhbin = np.arange(-1, 1.05, 0.05)
        ldbin = np.arange(-1, 1.05, 0.05)
        edges_neg = np.array([-0.3, -0.2, -0.1], dtype=np.float32)
        edges_dense = np.linspace(0.0, 0.8, 21, dtype=np.float32)
        edges_coarse = np.array([0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], np.float32)
        lvbin = np.round(np.concatenate([edges_neg, edges_dense, edges_coarse]), 6)

        lh_voxelcenter = (lhbin[:-1] + lhbin[1:]) / 2
        ld_voxelcenter = (ldbin[:-1] + ldbin[1:]) / 2
        lv_voxelcenter = (lvbin[:-1] + lvbin[1:]) / 2

        processed_data = []

        # === 3. 篩選資料 ===
        for frame in true_data2:
            filtered = [p for p in frame
                        if (x_min <= p[0] <= x_max and
                            y_min <= p[1] <= y_max and
                            z_min <= p[2] <= z_max)]
            filtered = np.array(filtered) if filtered else np.zeros((0, 3))
            processed_data.append(filtered)

        # === 4. 建立快取矩陣（前兩幀） ===
        # prev_xvoxel = np.zeros((x_size, z_size, 1))
        # prev_yvoxel = np.zeros((y_size, z_size, 1))
        # prev2_xvoxel = np.zeros((x_size, z_size, 1))
        # prev2_yvoxel = np.zeros((y_size, z_size, 1))

        # === 5. 主處理迴圈 ===
        for i, filtered in enumerate(processed_data):
            xvoxel = np.zeros((x_size, z_size, 1))
            yvoxel = np.zeros((y_size, z_size, 1))
            # xsvoxel = np.zeros((x_size, z_size, 1))
            # ysvoxel = np.zeros((y_size, z_size, 1))
            # co_xvoxel = np.zeros((x_size, z_size, 1))
            # co_yvoxel = np.zeros((y_size, z_size, 1))
            # === 5-1. 當幀累積 ===
            for point in filtered:
                x = point[0] - cluster_center[0]
                y = point[1] - cluster_center[1]
                z = point[2]
                # doppler = point[3]

                hlocal = np.argmin(np.abs(lh_voxelcenter - x))
                dlocal = np.argmin(np.abs(ld_voxelcenter - y))
                vlocal = np.argmin(np.abs(lv_voxelcenter - z))

                xvoxel[hlocal, vlocal, 0] += 1
                yvoxel[dlocal, vlocal, 0] += 1
                # xsvoxel[hlocal, vlocal, 0] += doppler
                # ysvoxel[dlocal, vlocal, 0] += doppler
                # co_xvoxel[hlocal, vlocal, 0] += x
                # co_yvoxel[dlocal, vlocal, 0] += y
            # === 5-2. 差分計算，直接用快取矩陣 ===
            # if i == 0:
            #     cxvoxel = np.zeros_like(xvoxel)
            #     cyvoxel = np.zeros_like(yvoxel)
            #     cxvoxel2 = np.zeros_like(xvoxel)
            #     cyvoxel2 = np.zeros_like(yvoxel)
            # elif i==1:
            #     cxvoxel = xvoxel - prev_xvoxel
            #     cyvoxel = yvoxel - prev_yvoxel
            #     cxvoxel2 = np.zeros_like(xvoxel)
            #     cyvoxel2 = np.zeros_like(yvoxel)
            # else:
            #     cxvoxel = xvoxel - prev_xvoxel
            #     cyvoxel = yvoxel - prev_yvoxel
            #     cxvoxel2 = xvoxel - prev2_xvoxel
            #     cyvoxel2 = yvoxel - prev2_yvoxel

            # === 5-3. 組合通道 ===
            voxel_frame = np.concatenate([
                xvoxel, yvoxel,
                # xsvoxel * 10, ysvoxel * 10,
                # cxvoxel, cyvoxel,
                # cxvoxel2, cyvoxel2,
                # co_xvoxel,co_yvoxel
            ], axis=2)

            # tvoxel.append(voxel_frame)
            if i == 0:
                tvoxel = voxel_frame
            else:
                tvoxel = np.concatenate([tvoxel, voxel_frame], axis=2)

            # === 5-4. 更新快取（保留前兩次） ===
            # prev2_xvoxel = prev_xvoxel
            # prev2_yvoxel = prev_yvoxel
            # prev_xvoxel = xvoxel
            # prev_yvoxel = yvoxel
        # last_voxel = np.asarray(tvoxel, dtype=np.float32)[-1]
        # return last_voxel
        return tvoxel