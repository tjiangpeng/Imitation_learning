import argparse
import logging
import sys
import os
from typing import Any
import pandas as pd
from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons

import tensorflow as tf

from argoData.tfrecord_config import convert_to_example
from hparms import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =============================================================================
# -- Constants ----------------------------------------------------------------
# =============================================================================
AGENT_TIME_STEP = 19
FIELD_VIEW = 80.0  # meter

FRAME_IN_SHARD = 512
# =============================================================================
# -- Functions ----------------------------------------------------------------
# =============================================================================


class ForecastingOnMapVisualizer:
    def __init__(
            self, dataset_dir: str, convert_tf_record=False, save_img=False, overwrite_rendered_file=True
    ) -> None:

        self.dataset_dir = dataset_dir
        print("Load data...")
        self.afl = ArgoverseForecastingLoader(dataset_dir)
        self.num = len(self.afl)
        self.log_agent_pose = None
        self.timestamps = None
        self.filenames = [None for i in range(self.num)]
        for i, seq in enumerate(self.afl.seq_list):
            self.filenames[i] = int(str(seq).split('/')[-1][0:-4])

        self.convert_tf_record = convert_tf_record
        self.overwrite_rendered_file = overwrite_rendered_file
        self.save_img = save_img

        if save_img:
            if not Path(f"{self.dataset_dir}../rendered_image").exists():
                os.makedirs(f"{self.dataset_dir}../rendered_image")
            num_shards = ceil(max(self.filenames) / FRAME_IN_SHARD)
            for i in range(1, num_shards+1):
                if not Path(f"{self.dataset_dir}../rendered_image/{i}").exists():
                    os.makedirs(f"{self.dataset_dir}../rendered_image/{i}")

    def plot_log_one_at_a_time(self, avm=None, log_num: int = 0):
        df = self.afl[log_num].seq_df
        city_name = df["CITY_NAME"].values[0]
        time_step = self.afl[log_num].agent_traj.shape[0]

        self.timestamps = list(sorted(set(df["TIMESTAMP"].values.tolist())))
        self.log_agent_pose = self.afl[log_num].agent_traj

        if not self.log_agent_pose.shape[0] == 50:
            sys.exit("The agent's tacking time is less than 5 seconds!")

        for i in range(time_step):
            if i < AGENT_TIME_STEP or i >= time_step - FUTURE_TIME_STEP:
                continue
            if self.save_img:
                file_num = self.filenames[log_num]
                shard_ind = ceil(file_num / FRAME_IN_SHARD)
                img_name = "{0}_{1:0>7d}_{2}.png".format(city_name, file_num, i)
                if (not self.overwrite_rendered_file) and \
                        Path(f"{self.dataset_dir}../rendered_image/{shard_ind}/{img_name}").exists():
                    continue

            [xcenter, ycenter] = self.afl[log_num].agent_traj[i, :]

            # Figure setting
            fig = plt.figure(num=1, figsize=(4.3, 4.3), facecolor='k')
            plt.clf()
            plt.axis("off")
            ax = fig.add_subplot(111)
            ax.set_xlim([-FIELD_VIEW/2+xcenter, FIELD_VIEW/2+xcenter])
            ax.set_ylim([-FIELD_VIEW/2+ycenter, FIELD_VIEW/2+ycenter])
            fig.tight_layout()

            # Draw map
            xmin = xcenter - 80
            xmax = xcenter + 80
            ymin = ycenter - 80
            ymax = ycenter + 80
            local_lane_polygons = avm.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
            local_das = avm.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
            draw_lane_polygons(ax, local_lane_polygons, color="tab:gray")
            draw_lane_polygons(ax, local_das, color=(1, 0, 1))

            # Render and get the log info
            viz = False
            if self.save_img:
                viz = True
            if self.convert_tf_record:
                # Save the map image
                plt.savefig("temp.png", dpi=100, facecolor='k', bbox_inches='tight', pad_inches=0)

            past_traj = self.get_agent_past_traj(ax, i, viz)
            future_traj = self.get_agent_future_traj(ax, i, viz)
            center_lines = self.get_candidate_centerlines(ax, self.log_agent_pose[:(AGENT_TIME_STEP+1), :],
                                                          avm, city_name, viz)
            surr_past_pos = self.get_surr_past_traj(ax, df, i, self.log_agent_pose[i, :], viz)

            if self.save_img:
                shard_ind = 1
                plt.savefig(
                    f"{self.dataset_dir}../rendered_image/{shard_ind}/{img_name}",
                    dpi=100, facecolor='k', bbox_inches='tight', pad_inches=0)

        return past_traj, future_traj, center_lines, surr_past_pos

    def get_surr_past_traj(
        self,
        ax: Axes,
        df: pd.DataFrame,
        cur_time: int,
        agent_pos,
        viz=False
    ) -> None:
        which_timestamps = self.timestamps[cur_time-SURR_TIME_STEP:cur_time+1]

        # Remove agent row
        df = df[df["OBJECT_TYPE"] != "AGENT"]
        frames = df.groupby("TIMESTAMP")
        # get dataset at "which_timestamps"
        which_groups = [None for i in range(0, SURR_TIME_STEP+1)]
        for group_name, group_data in frames:
            if group_name == which_timestamps[-1]:
                track_ids = group_data["TRACK_ID"].values.tolist()
            if group_name in which_timestamps:
                ind = which_timestamps.index(group_name)
                which_groups[ind] = group_data.values[:, 1:5]

        surr_past_pos = np.ones([len(track_ids), (SURR_TIME_STEP+1)*2]) * 10000.0
        # render color with decreasing lightness
        color_lightness = np.linspace(1, 0, SURR_TIME_STEP+2)[1:].tolist()
        marker_size = 10
        for idx, group in enumerate(which_groups):
            for i in range(group.shape[0]):
                if group[i, 0] in track_ids:
                    x = group[i, 2]
                    y = group[i, 3]

                    surr_past_pos[track_ids.index(group[i, 0]), idx] = x - agent_pos[0]
                    surr_past_pos[track_ids.index(group[i, 0]), idx+SURR_TIME_STEP+1] = y - agent_pos[1]
                    if viz:
                        c = [(1, 1, color_lightness[idx])]
                        if idx == SURR_TIME_STEP:
                            marker_size = 20
                        ax.scatter(x, y, s=marker_size, c=c, alpha=1, zorder=2)

        return surr_past_pos

    def get_agent_past_traj(
        self,
        ax: Axes,
        cur_time: int,
        viz=False
    ):
        if viz:
            marker_size = 10
            color_lightness = np.linspace(1, 0, AGENT_TIME_STEP + 1)[1:].tolist()
            for i in range(cur_time-AGENT_TIME_STEP, cur_time+1):
                if i == cur_time:
                    marker_size = 10
                    color = [(0, 1, 0)]
                else:
                    color = [(color_lightness[i], color_lightness[i], 1)]
                x = self.log_agent_pose[i, 0]
                y = self.log_agent_pose[i, 1]
                ax.scatter(x, y, s=marker_size, c=color, alpha=1, zorder=4)

        past_traj = self.log_agent_pose[cur_time-AGENT_TIME_STEP:cur_time, :] - self.log_agent_pose[cur_time, :]
        return past_traj

    def get_candidate_centerlines(
        self,
        ax: Axes,
        agent_obs_traj: np.ndarray,
        avm,
        city_name: str,
        viz=False
    ):
        candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, city_name)
        for ind, centerline in enumerate(candidate_centerlines):
            if viz:
                ax.plot(centerline[:, 0].tolist(), centerline[:, 1].tolist(), "-", color="w", zorder=3)
                # ax.scatter(centerline[:, 0].tolist(), centerline[:, 1].tolist(), s=10, color="w", zorder=3)
            # get the coordinate relative to agent pos
            candidate_centerlines[ind] = centerline - agent_obs_traj[-1, :]

        return candidate_centerlines

    def get_agent_future_traj(
        self,
        ax: Axes,
        cur_time: int,
        viz=False
    ):
        traj = self.log_agent_pose[cur_time+1:cur_time+FUTURE_TIME_STEP+1, :]
        if viz:
            for t in traj:
                ax.scatter(t[0], t[1], s=2, c='r')
        relative_traj = traj - self.log_agent_pose[cur_time, :]
        return relative_traj


def visualize_forecating_data_on_map(args: Any) -> None:
    print("Loading map...")
    avm = ArgoverseMap()
    fomv = ForecastingOnMapVisualizer(dataset_dir=args.dataset_dir,
                                      save_img=args.save_image,
                                      overwrite_rendered_file=args.overwrite_rendered_file)
    for i in range(fomv.num):
        print(f"Processing the file: {fomv.filenames[i]}")
        fomv.plot_log_one_at_a_time(avm, log_num=i)


def write_tf_record(args: Any) -> None:
    print("Loading map...")
    avm = ArgoverseMap()
    fomv = ForecastingOnMapVisualizer(dataset_dir=args.dataset_dir,
                                      convert_tf_record=True,
                                      overwrite_rendered_file=args.overwrite_rendered_file)

    if not Path(f"{args.dataset_dir}../tf_record").exists():
        os.makedirs(f"{args.dataset_dir}../tf_record")

    print("Start rendering...")
    for i in range(args.starting_frame_ind, fomv.num):
    # for i in range(9216, 51200):
        if (i+1) % 64 == 0:
            print(f"Processing {i}th frame")
        if i % FRAME_IN_SHARD == 0:
            shard_ind = int(i / FRAME_IN_SHARD)
            writer = tf.io.TFRecordWriter(f"{args.dataset_dir}/../tf_record/{shard_ind}_tf_record")

        past_traj, future_traj, center_lines, surr_past_pos = fomv.plot_log_one_at_a_time(avm, log_num=i)
        example = convert_to_example(past_traj, future_traj, center_lines, surr_past_pos)
        writer.write(example.SerializeToString())

        if (i-shard_ind*FRAME_IN_SHARD) % (FRAME_IN_SHARD-1) == 0 and (not i == shard_ind*FRAME_IN_SHARD):
            print(f"the {shard_ind}th shard has been done!")
            writer.close()
    writer.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="path to where the logs live",
                        default="../../data/argo/forecasting/sample/data/")
    parser.add_argument("--convert_tf_record", help="convert to tfrecord file or not",
                        default=True)
    parser.add_argument("--starting_frame_ind", type=int, help="which frame to start",
                        default=0)
    parser.add_argument("--save_image", help="save rendered image or not",
                        default=False)
    parser.add_argument("--overwrite_rendered_file", help="overwrite the rendered files or not",
                        default=True)

    args = parser.parse_args()
    logger.info(args)

    if args.convert_tf_record:
        write_tf_record(args)
    else:
        visualize_forecating_data_on_map(args)


if __name__ == '__main__':
    main()

