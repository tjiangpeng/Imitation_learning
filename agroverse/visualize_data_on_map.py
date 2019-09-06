# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import argparse
import copy
import glob
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio
# all mayavi imports MUST come before matplotlib, else Tkinter exceptions
# will be thrown, e.g. "unrecognized selector sent to instance"
import mayavi
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.camera_stats import RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.cuboid_interior import filter_point_cloud_to_bbox_2D_vectorized
from argoverse.utils.ffmpeg_utils import write_nonsequential_idx_video, write_video
from argoverse.utils.geometry import filter_point_cloud_to_polygon, rotate_polygon_about_pt
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_bbox_2D
from argoverse.utils.pkl_utils import load_pkl_dictionary
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.visualization.ground_visualization import draw_ground_pts_in_image
from argoverse.visualization.mayavi_utils import draw_lidar
from argoverse.visualization.mpl_point_cloud_vis import draw_point_cloud_bev

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

IS_OCCLUDED_FLAG = 100
LANE_TANGENT_VECTOR_SCALING = 4

HEAD_DIST = 60.0
FIELD_VIEW = 80.0

EGO_TIME_STEP = 20
SURR_TIME_STEP = 10
FUTURE_TIME_STEP = 10

"""
Code to plot track label trajectories on a map, for the tracking benchmark.
"""


class DatasetOnMapVisualizer:
    def __init__(
        self, dataset_dir: str, experiment_prefix: str, use_existing_files: bool = True, log_id: str = None
    ) -> None:
        """We will cache the accumulated trajectories per city, per log, and per frame
        for the tracking benchmark.
            """
        self.plot_lane_tangent_arrows = True
        self.plot_lidar_bev = True
        self.plot_lidar_in_img = False
        self.experiment_prefix = experiment_prefix
        self.dataset_dir = dataset_dir
        self.labels_dir = dataset_dir
        self.sdb = SynchronizationDB(self.dataset_dir)

        if log_id is None:
            tmp_dir = tempfile.gettempdir()
            per_city_traj_dict_fpath = f"{tmp_dir}/per_city_traj_dict_{experiment_prefix}.pkl"
            log_egopose_dict_fpath = f"{tmp_dir}/log_egopose_dict_{experiment_prefix}.pkl"
            log_timestamp_dict_fpath = f"{tmp_dir}/log_timestamp_dict_{experiment_prefix}.pkl"
            if not use_existing_files:
                # write the accumulate data dictionaries to disk
                PerFrameLabelAccumulator(dataset_dir, dataset_dir, experiment_prefix)

            self.per_city_traj_dict = load_pkl_dictionary(per_city_traj_dict_fpath)
            self.log_egopose_dict = load_pkl_dictionary(log_egopose_dict_fpath)
            self.log_timestamp_dict = load_pkl_dictionary(log_timestamp_dict_fpath)
        else:
            pfa = PerFrameLabelAccumulator(dataset_dir, dataset_dir, experiment_prefix, save=False)
            pfa.accumulate_per_log_data(log_id=log_id)
            self.per_city_traj_dict = pfa.per_city_traj_dict
            self.log_egopose_dict = pfa.log_egopose_dict
            self.log_timestamp_dict = pfa.log_timestamp_dict

            self.timestamps = []
            fpaths = sorted(glob.glob(f"{self.dataset_dir}/{log_id}/per_sweep_annotations_amodal/tracked_object_labels_*.json"))
            for i, fpath in enumerate(fpaths):
                stamp = fpath.split("/")[-1].split(".")[0].split("_")[-1]
                self.timestamps.append(int(stamp))

    def plot_log_one_at_a_time(self, log_id="", idx=-1, save_video=True, city=""):
        """
        Playback a log in the static context of a map.
        In the far left frame, we show the car moving in the map. In the middle frame, we show
        the car's LiDAR returns (in the map frame). In the far right frame, we show the front camera's
        RGB image.
        """
        avm = ArgoverseMap()
        for city_name, trajs in self.per_city_traj_dict.items():
            if city != "":
                if city != city_name:
                    continue
            if city_name not in ["PIT", "MIA"]:
                logger.info("Unknown city")
                continue

            log_ids = []
            logger.info(f"{city_name} has {len(trajs)} tracks")

            if log_id == "":
                # first iterate over the instance axis
                for traj_idx, (traj, log_id) in enumerate(trajs):
                    log_ids += [log_id]
            else:
                log_ids = [log_id]

            # eliminate the duplicates
            for log_id in set(log_ids):
                logger.info(f"Log: {log_id}")

                # then iterate over the time axis
                for i, timestamp in enumerate(self.timestamps):
                    if i < EGO_TIME_STEP:
                        continue
                    if i > len(self.timestamps) - FUTURE_TIME_STEP:
                        break
                    if idx != -1:
                        if i != idx:
                            continue
                    if i % 500 == 0:
                        logger.info(f"\tOn file {i} of {log_id}")

                    if timestamp not in self.log_egopose_dict[log_id]:
                        all_available_timestamps = sorted(self.log_egopose_dict[log_id].keys())
                        diff = (all_available_timestamps[0] - timestamp) / 1e9
                        logger.info(f"{diff:.2f} sec before first labeled sweep")
                        continue

                    logger.info(f"\tt={timestamp}")
                    if self.plot_lidar_bev:
                        fig = plt.figure(figsize=(4, 4), facecolor='k')
                        # plt.title(f"Log {log_id} @t={lidar_timestamp} in {city_name}")
                        plt.axis("off")
                        # ax_map = fig.add_subplot(131)
                        ax_3d = fig.add_subplot(111)
                        # ax_3d.set_facecolor("black")
                        # ax_rgb = fig.add_subplot(133)
                        ax_3d.set_xlim([-FIELD_VIEW + HEAD_DIST, HEAD_DIST])
                        ax_3d.set_ylim([-FIELD_VIEW/2, FIELD_VIEW/2])

                    # need the ego-track here
                    pose_city_to_ego = self.log_egopose_dict[log_id][timestamp]
                    xcenter = pose_city_to_ego["translation"][0]
                    ycenter = pose_city_to_ego["translation"][1]
                    ego_center_xyz = np.array(pose_city_to_ego["translation"])

                    city_to_egovehicle_se3 = SE3(rotation=pose_city_to_ego["rotation"], translation=ego_center_xyz)

                    if self.plot_lidar_bev:
                        xmin = xcenter - 80  # 150
                        xmax = xcenter + 80  # 150
                        ymin = ycenter - 80  # 150
                        ymax = ycenter + 80  # 150
                        # ax_map.scatter(xcenter, ycenter, 200, color="g", marker=".", zorder=2)
                        # ax_map.set_xlim([xmin, xmax])
                        # ax_map.set_ylim([ymin, ymax])
                        local_lane_polygons = avm.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
                        local_das = avm.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)


                    if self.plot_lidar_bev:

                        self.render_bev_labels_mpl(
                            city_name,
                            ax_3d,
                            "ego_axis",
                            # lidar_pts,
                            copy.deepcopy(local_lane_polygons),
                            copy.deepcopy(local_das),
                            log_id,
                            timestamp,
                            i,
                            city_to_egovehicle_se3,
                            avm,
                        )

                        self.render_ego_past_traj(
                            ax_3d,
                            log_id,
                            i,
                            city_to_egovehicle_se3
                        )

                        fig.tight_layout()
                        if not Path(f"{self.experiment_prefix}_per_log_viz/{log_id}").exists():
                            os.makedirs(f"{self.experiment_prefix}_per_log_viz/{log_id}")

                        plt.savefig(
                            f"{self.experiment_prefix}_per_log_viz/{log_id}/{city_name}_{log_id}_{timestamp}.png",
                            dpi=100, facecolor='k'
                        )
                    # plt.show()
                    # plt.close("all")

                # after all frames are processed, write video with saved images
                if save_video:
                    if self.plot_lidar_bev:
                        fps = 10
                        img_wildcard = f"{self.experiment_prefix}_per_log_viz/{log_id}/{city_name}_{log_id}_%*.png"
                        output_fpath = f"{self.experiment_prefix}_per_log_viz/{log_id}_lidar_roi_nonground.mp4"
                        write_nonsequential_idx_video(img_wildcard, output_fpath, fps)

                    if self.plot_lidar_in_img:
                        for camera_name in RING_CAMERA_LIST + STEREO_CAMERA_LIST:
                            image_prefix = (
                                f"{self.experiment_prefix}_per_log_viz/{log_id}/{camera_name}/{camera_name}_%d.jpg"
                            )
                            output_prefix = f"{self.experiment_prefix}_per_log_viz/{log_id}_{camera_name}"
                            write_video(image_prefix, output_prefix)

    def render_bev_labels_mpl(
        self,
        city_name: str,
        ax: plt.Axes,
        axis: str,
        # lidar_pts: np.ndarray,
        local_lane_polygons: np.ndarray,
        local_das: np.ndarray,
        log_id: str,
        timestamp: int,
        timestamp_ind: int,
        city_to_egovehicle_se3: SE3,
        avm: ArgoverseMap,
    ) -> None:
        """Plot nearby lane polygons and nearby driveable areas (da) on the Matplotlib axes.

        Args:
            city_name: The name of a city, e.g. `"PIT"`
            ax: Matplotlib axes
            axis: string, either 'ego_axis' or 'city_axis' to demonstrate the
            lidar_pts:  Numpy array of shape (N,3)
            local_lane_polygons: Polygons representing the local lane set
            local_das: Numpy array of objects of shape (N,) where each object is of shape (M,3)
            log_id: The ID of a log
            timestamp: In nanoseconds
            city_to_egovehicle_se3: Transformation from egovehicle frame to city frame
            avm: ArgoverseMap instance
        """
        if axis is not "city_axis":
            # rendering instead in the egovehicle reference frame
            for da_idx, local_da in enumerate(local_das):
                local_da = city_to_egovehicle_se3.inverse_transform_point_cloud(local_da)
                # local_das[da_idx] = rotate_polygon_about_pt(local_da, city_to_egovehicle_se3.rotation, np.zeros(3))
                local_das[da_idx] = local_da

            for lane_idx, local_lane_polygon in enumerate(local_lane_polygons):
                local_lane_polygon = city_to_egovehicle_se3.inverse_transform_point_cloud(local_lane_polygon)
                # local_lane_polygons[lane_idx] = rotate_polygon_about_pt(
                #     local_lane_polygon, city_to_egovehicle_se3.rotation, np.zeros(3)
                # )
                local_lane_polygons[lane_idx] = local_lane_polygon

        draw_lane_polygons(ax, local_lane_polygons, color="tab:gray")
        draw_lane_polygons(ax, local_das, color=(1, 0, 1))

        # render ego vehicle
        bbox_ego = np.array([[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
        # bbox_ego = rotate_polygon_about_pt(bbox_ego, city_to_egovehicle_se3.rotation, np.zeros((3,)))
        plot_bbox_2D(ax, bbox_ego, 'g')

        # render surrounding vehicle and pedestrians
        objects = self.log_timestamp_dict[log_id][timestamp]
        hist_objects = {}

        all_occluded = True
        for frame_rec in objects:
            if frame_rec.occlusion_val != IS_OCCLUDED_FLAG:
                all_occluded = False
            hist_objects[frame_rec.track_uuid] = [None for i in range(0, SURR_TIME_STEP+1)]
            hist_objects[frame_rec.track_uuid][SURR_TIME_STEP] = frame_rec.bbox_city_fr

        if not all_occluded:
            for ind, i in enumerate(range(timestamp_ind-SURR_TIME_STEP, timestamp_ind)):
                objects = self.log_timestamp_dict[log_id][self.timestamps[i]]
                for frame_rec in objects:
                    if frame_rec.track_uuid in hist_objects:
                        hist_objects[frame_rec.track_uuid][ind] = frame_rec.bbox_city_fr

            # render color with decreasing lightness
            color_lightness = np.linspace(1, 0, SURR_TIME_STEP+2)[1:].tolist()
            for track_id in hist_objects:
                for ind_color, bbox_city_fr in enumerate(hist_objects[track_id]):
                    if bbox_city_fr is None:
                        continue
                    bbox_relative = city_to_egovehicle_se3.inverse_transform_point_cloud(bbox_city_fr)

                    x = bbox_relative[:, 0].tolist()
                    y = bbox_relative[:, 1].tolist()
                    x[1], x[2], x[3], y[1], y[2], y[3] = x[2], x[3], x[1], y[2], y[3], y[1]
                    ax.fill(x, y, c=(1, 1, color_lightness[ind_color]), alpha=1)

        else:
            logger.info(f"all occluded at {timestamp}")
            xcenter = city_to_egovehicle_se3.translation[0]
            ycenter = city_to_egovehicle_se3.translation[1]
            ax.text(xcenter - 146, ycenter + 50, "ALL OBJECTS OCCLUDED", fontsize=30)

    def render_ego_past_traj(
        self,
        axes: Axes,
        log_id: str,
        timestamp_ind: int,
        city_to_egovehicle_se3: SE3,
    ) -> None:

        x = [0]
        y = [0]
        for i in range(timestamp_ind-EGO_TIME_STEP, timestamp_ind):
            timestamp = self.timestamps[i]
            pose_city_to_ego = self.log_egopose_dict[log_id][timestamp]
            relative_pose = city_to_egovehicle_se3.inverse_transform_point_cloud(pose_city_to_ego["translation"])
            x.append(relative_pose[0])
            y.append(relative_pose[1])
        axes.scatter(x, y, s=5, c='b')


def visualize_30hz_benchmark_data_on_map(args: Any) -> None:
    """
    """
    domv = DatasetOnMapVisualizer(
        args.dataset_dir, args.experiment_prefix, log_id=args.log_id, use_existing_files=args.use_existing_files
    )
    # Plotting does not work on AWS! The figure cannot be refreshed properly
    # Thus, plotting must be performed locally.
    domv.plot_log_one_at_a_time()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="path to where the logs live",
                        default='../../data/argo/argoverse-tracking/sample')
    parser.add_argument(
        "--experiment_prefix",
        default="argoverse_bev_viz",
        type=str,
        help="results will be saved in a folder with this prefix for its name",
    )
    parser.add_argument(
        "--log_id", default='c6911883-1843-3727-8eaa-41dc8cda8993', type=str,
        help="log ids, this is the folder name in argoverse-tracking/*/[log_id]",
    )

    parser.add_argument(
        "--use_existing_files",
        action="store_true",
        help="load pre-saved log data from pkl dictionaries instead of scraping it",
    )
    args = parser.parse_args()
    logger.info(args)
    visualize_30hz_benchmark_data_on_map(args)
