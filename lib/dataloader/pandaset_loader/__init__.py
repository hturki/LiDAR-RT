# from pathlib import Path
# from typing import Dict
# import numpy as np
# import torch
# from lib.scene.bounding_box import BoundingBox
# from lib.scene.lidar_sensor import LiDARSensor
# from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSet, PandaSetDataParserConfig
# from lib.utils.general_utils import matrix_to_quaternion


def load_pandaset_raw(base_dir, args):
    config = PandaSetDataParserConfig(data=Path(base_dir).parent, sequence=Path(base_dir).name, add_missing_points=False, val_every_n_frames=0, correct_cuboid_time=False)    
    panda_set = PandaSet(config)
    dataparser_outputs = panda_set._generate_dataparser_outputs()

    inclination_bounds = [np.radians(config.lidar_elevation_mapping["Pandar64"][i]) for i in range(len(config.lidar_elevation_mapping["Pandar64"]))]
    lidar = LiDARSensor(
        sensor2ego=np.eye(4),
        name="pandar64",
        inclination_bounds=inclination_bounds,
        data_type=args.data_type,
    )

    W, H = 1800, 64
    lidar_metadata = dataparser_outputs.metadata["lidars"]
    lidar_points = dataparser_outputs.metadata["point_clouds"]
    num_frames = len(lidar_metadata.lidar_to_worlds)
    for frame_index in range(num_frames):
        xyzs, intensities = lidar_points[frame_index][:, :3], lidar_points[frame_index][:, 3]

        range_map = np.ones((H, W)) * -1
        intensity_map = np.ones((H, W)) * -1

        try:
            h_indices = panda_set._add_channel_info(lidar_points[frame_index], -1, "Pandar64")[..., -2].int()
            w_indices = ((np.arctan2(xyzs[..., 1], xyzs[..., 0]) + np.pi) / np.radians(0.2)).int()
            dists = np.linalg.norm(lidar_points[frame_index][:, :3], axis=1)

            for intensity, dist, h_idx, w_idx in zip(intensities, dists, h_indices, w_indices):
                if (w_idx < 0) or (w_idx >= W) or (h_idx < 0) or (h_idx >= H):
                    continue

                if range_map[h_idx, w_idx] == -1:
                    range_map[h_idx, w_idx] = dist
                    intensity_map[h_idx, w_idx] = intensity
                elif range_map[h_idx, w_idx] > dist:
                    range_map[h_idx, w_idx] = dist
                    intensity_map[h_idx, w_idx] = intensity
        except:
            print("Skipping frame", frame_index)
        
        range_image_r1 = np.stack([range_map, intensity_map], axis=-1)
        range_image_r2 = np.ones_like(range_image_r1) * -1
        
        ego2world = torch.eye(4)
        ego2world[:3] = lidar_metadata.lidar_to_worlds[frame_index]
        print((range_image_r1[..., 0] > 0).sum())
        lidar.add_frame(
            frame=frame_index+1, ego2world=ego2world, r1=range_image_r1, r2=range_image_r2
        )

    bboxes: Dict[str, BoundingBox] = {}  # frame * n
    actor_trajectories = panda_set._get_actor_trajectories()
    for actor_trajectory in actor_trajectories:
        if actor_trajectory["stationary"]:
            continue

        actor_timestamps = torch.DoubleTensor(actor_trajectory["timestamps"])

        object_id = str(len(bboxes))
        bboxes[object_id] = BoundingBox(1, str(len(bboxes)), actor_trajectory["dims"])

        for frame_idx in range(num_frames):
            pose_idx = (actor_timestamps - panda_set.sequence.camera["front_camera"].timestamps[frame_idx]).abs().argmin()
            pose = actor_trajectory["poses"][pose_idx]
            pos = pose[:3, 3]
            quaternion = matrix_to_quaternion(pose[:3, :3])
            quaternion = quaternion / torch.norm(quaternion)
            quaternion = quaternion.unsqueeze(0)
            dT = torch.zeros(3).float().cuda()
            dR = torch.eye(3).float().cuda()
            bboxes[object_id].frame[frame_idx + 1] = (pos.cuda(), quaternion.cuda(), dT, dR)

    return lidar, bboxes
