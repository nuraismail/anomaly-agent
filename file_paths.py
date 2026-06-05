from pathlib import Path

sim_maps_file = "CMBmapsPlanckLCDM256.npy"
planck_map_file = "COM_CMB_IQU-smica_2048_R3.00_full.fits"

cmb_dict_file = "cmb_dict.yaml"

plot_config_file = "plot_config.yaml"
test_config_file = "test_config.yaml"
agent_config_file = "agent_config.yaml"

planner_file = "planner.yaml"
implement_file = "implement.yaml"
hypothesis_file = "hypothesis.yaml"
summary_file = "summary.yaml"

data_dir = "data"
data_dir = Path(data_dir)

input_dir = data_dir / "input"

output_dir = data_dir / "output"

checkpoint_db_path = data_dir / "anomaly_agent.db"

configs_dir = "configs"
configs_dir = Path(configs_dir)

prompts_dir = configs_dir / "prompts"

sim_maps_path = input_dir / sim_maps_file
planck_map_path = input_dir / planck_map_file

cmb_dict_dir = configs_dir / cmb_dict_file

plot_config_dir = configs_dir / plot_config_file
test_config_dir = configs_dir / test_config_file
agent_config_dir = configs_dir / agent_config_file

planner_dir = prompts_dir / planner_file
implement_dir = prompts_dir / implement_file
hypothesis_dir = prompts_dir / hypothesis_file
summary_dir = prompts_dir / summary_file
