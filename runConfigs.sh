#!/bin/bash

# Script for running experiments continuously.
# Takes config files from the /Experiments/ConfigFiles directory, moves them
# to the working directory, runs the experiment (which generates all the
# necessary directories itself), then moves the config to the experiment's
# directory and grabs the next one, repeating until ConfigFiles is empty.

# While ConfigFiles is not empty
while [ -n "$(ls Experiments/ConfigQueue)" ]; do
	# Move first available file to the working directory
	fpath=$(echo Experiments/ConfigQueue/*.py)	# Gets all file paths
	fpath=${fpath%% *}				# Gets first file path
	fname=${fpath##*/}				# Gets file name
	mv "$fpath" "config.py"				# Move to pwd
	# Run train.py (this generates experiment directory)
	echo "Training ${fname}"
	python3 train.py
	# Move config into generated directory
	expName=$(python3 -c "import config; print(config.Config().NAME)")	# Get experiment name from python file
	mv "config.py" "Experiments/${expName}/config_${expName}"
	echo "Done training ${expName}"
done
# Print "done"?
echo "All experiments finished training"
