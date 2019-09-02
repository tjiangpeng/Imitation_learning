from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.map_representation.map_api import ArgoverseMap

##set root_dir to the correct path to your dataset folder
root_dir = 'forecasting_sample/data/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:', len(afl))

for argoverse_forecasting_data in (afl):
    print(argoverse_forecasting_data)
    print(argoverse_forecasting_data.track_id_list)

seq_path = f"{root_dir}/2645.csv"
viz_sequence(afl.get(seq_path).seq_df, show=True)


avm = ArgoverseMap()

obs_len = 20

seq_path = f"{root_dir}/2645.csv"
agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[1].city, viz=True)


a = 1