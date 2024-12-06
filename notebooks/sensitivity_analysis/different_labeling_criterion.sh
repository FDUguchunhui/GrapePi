#!/bin/zsh

config_name=("grapepi_sageconv" "grapepi_gcnconv")
label_col=("hard_label_02_08" "hard_label_03_07" "hard_label_04_06")

for config in "${config_name[@]}"; do
    for label in "${label_col[@]}"; do
        run_name="${config}_label_${label}"
        command="python main.py --config-path configs/protein --config-name $config dataset.dir=data/gastric_all_data dataset.label_col=$label dataset.interaction_conf_col=combined_score dataset.interaction_conf_thresh=400.0 'dataset.node_numeric_cols=[protein_probability]' train.early_stop=True run.name=$run_name"
        echo "Running command: $command"
        eval $command
    done
done