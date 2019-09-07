import argparse
import logging
import glob
import sys
import os
from typing import Any
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =============================================================================
# -- Constants ----------------------------------------------------------------
# =============================================================================
EGO_TIME_STEP = 19
SURR_TIME_STEP = 10
FUTURE_TIME_STEP = 30

FIELD_VIEW = 80.0  # meter
# =============================================================================
# -- Functions ----------------------------------------------------------------
# =============================================================================


class ForecastingOnMapVisualizer:
    def __init__(
            self, dataset_dir: str
    ) -> None:

        self.dataset_dir = dataset_dir
        self.afl = ArgoverseForecastingLoader(dataset_dir)
        self.num = len(self.afl)
        self.log_agent_pose = None
        self.timestamps = None

    def plot_log_one_at_a_time(self, avm=None, log_num: int = 0, save_img=False):
        df = self.afl[log_num].seq_df
        city_name = df["CITY_NAME"].values[0]
        time_step = self.afl[log_num].agent_traj.shape[0]

        self.timestamps = list(sorted(set(df["TIMESTAMP"].values.tolist())))
        self.log_agent_pose = self.afl[log_num].agent_traj
        # hist_objects[track_id] = [0,0,0,0,0,....]

        frames = df.groupby("TIMESTAMP")
        for group_name, group_data in frames:
            print(group_name)
            pass

        for i in range(time_step):
            if i < EGO_TIME_STEP or i >= time_step - FUTURE_TIME_STEP:
                continue
            [xcenter, ycenter] = self.afl[log_num].agent_traj[i, :]

            # Figure setting
            fig = plt.figure(num=1, figsize=(4.3, 4.3), facecolor='k')
            plt.axis("off")
            ax = fig.add_subplot(111)
            ax.set_xlim([-FIELD_VIEW/2+xcenter, FIELD_VIEW/2+xcenter])
            ax.set_ylim([-FIELD_VIEW/2+ycenter, FIELD_VIEW/2+ycenter])

            # Draw map
            xmin = xcenter - 80
            xmax = xcenter + 80
            ymin = ycenter - 80
            ymax = ycenter + 80
            local_lane_polygons = avm.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
            local_das = avm.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)
            draw_lane_polygons(ax, local_lane_polygons, color="tab:gray")
            draw_lane_polygons(ax, local_das, color=(1, 0, 1))

            self.render_surr_past_traj(ax, df, i)
            self.render_agent_past_traj(ax, i)
            traj = self.get_future_traj(ax, i, True)

            fig.tight_layout()
            if save_img:
                if not Path(f"{self.dataset_dir}../rendered_image").exists():
                    os.makedirs(f"{self.dataset_dir}../rendered_image")
                plt.savefig(
                    f"{self.dataset_dir}../rendered_image/{city_name}_{log_num}_{i}.png",
                    dpi=100, facecolor='k', bbox_inches='tight', pad_inches=0
                )

    def render_surr_past_traj(
        self,
        ax: Axes,
        df: pd.DataFrame,
        cur_time: int
    ) -> None:
        which_timestamps = self.timestamps[cur_time-SURR_TIME_STEP:cur_time+1]

        frames = df.groupby("TIMESTAMP")
        # get dataset at "which_timestamps"
        which_groups = [None for i in range(0, SURR_TIME_STEP+1)]
        for group_name, group_data in frames:
            if group_name == which_timestamps[-1]:
                track_ids = group_data["TRACK_ID"].values
                # x = group_data["X"].values
                # y = group_data["Y"].values
                # for idx, track_id in enumerate(group_data["TRACK_ID"].values):
                #     hist_objects[track_id] = [None for i in range(0, SURR_TIME_STEP+1)]
                #     hist_objects[track_id][SURR_TIME_STEP] = [x[idx], y[idx]]
            if group_name in which_timestamps:
                ind = which_timestamps.index(group_name)
                which_groups[ind] = group_data.values[:, 1:5]

        # render color with decreasing lightness
        color_lightness = np.linspace(1, 0, SURR_TIME_STEP+2)[1:].tolist()
        marker_size = 2
        for idx, group in enumerate(which_groups):
            if idx == SURR_TIME_STEP:
                marker_size = 10
            for i in range(group.shape[0]):
                if not group[i, 1] == 'AGENT':
                # if group[i, 1] == 'OTHERS':
                    c = [(1, 1, color_lightness[idx])]
                # elif group[i, 1] == 'AV':
                #     c = [(color_lightness[idx], 1, color_lightness[idx])]
                else:
                    continue
                if group[i, 0] in track_ids:
                    x = group[i, 2]
                    y = group[i, 3]
                    ax.scatter(x, y, s=marker_size, c=c, alpha=1)

    def render_agent_past_traj(
        self,
        ax: Axes,
        cur_time: int
    ):
        marker_size = 2
        color = 'b'
        for i in range(cur_time-EGO_TIME_STEP, cur_time+1):
            if i == cur_time:
                marker_size = 10
                color = 'g'
            x = self.log_agent_pose[i, 0]
            y = self.log_agent_pose[i, 1]
            ax.scatter(x, y, s=marker_size, c=color, alpha=1)

        past_traj = self.log_agent_pose[cur_time-EGO_TIME_STEP:cur_time, :] - self.log_agent_pose[cur_time, :]
        return past_traj

    def get_future_traj(
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
    avm = ArgoverseMap()

    fomv = ForecastingOnMapVisualizer(args.dataset_dir)
    for i in range(fomv.num):
        fomv.plot_log_one_at_a_time(avm, log_num=i, save_img=args.save_image)
    a = 1


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="path to where the logs live",
                        default="../../data/argo/forecasting/sample/data/")
    parser.add_argument("--save_image", help="save rendered image or not",
                        default=True)

    args = parser.parse_args()
    logger.info(args)
    visualize_forecating_data_on_map(args)


if __name__ == '__main__':
    main()

