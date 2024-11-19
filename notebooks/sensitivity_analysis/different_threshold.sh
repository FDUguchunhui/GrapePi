#!/bin/zsh

# Define the different threshold values
thresholds=(150.0 400.0 700.0 900.0)

# Define the different config names
config_names=("grapepi_sageconv" "grapepi_gcnconv")

# Loop through each config name and threshold value and run the Python script
for config_name in $config_names; do
  for threshold in $thresholds; do
    run_name="${config_name}_thresh_${threshold}"
    python main.py --config-path configs/protein --config-name $config_name \
      dataset.interaction_conf_col=combined_score \
      dataset.interaction_conf_thresh=$threshold \
      'dataset.node_numeric_cols=[protein_probability]' \
      train.early_stop=True \
      run.name=$run_name
  done
done