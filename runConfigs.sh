#!/bin/bash

# Script for running experiments continuously.
# Takes config files from the /Experiments/ConfigFiles directory, moves them
# to the working directory, runs the experiment (which generates all the
# necessary directories itself), then moves the config to the experiment's
# directory and grabs the next one, repeating until ConfigFiles is empty.

echo "Beginning training" > train.log
# While ConfigFiles is not empty
while [ -n "$(ls Experiments/ConfigQueue)" ]; do
	# Move first available file to the working directory
	fpath=$(echo Experiments/ConfigQueue/*.py)	# Gets all file paths
	fpath=${fpath%% *}				# Gets first file path
	fname=${fpath##*/}				# Gets file name
	echo "fpath: ${fpath}" >> train.log
	echo "fname: ${fname}" >> train.log
	mv "$fpath" "config.py"				# Move to pwd
	# Run train.py (this generates experiment directory)
	echo "Training ${fname}" >> train.log
	echo "Training ${fname}"
	expName=$(python3 -B -c "import config; print(config.Config().NAME)")	# Get experiment name from python file
	expPath=$(python3 -B -c "import config; print(config.Config().SAVE_PATH)")	# Get experiment save path from python file
	echo "expName: ${expName}" >> train.log
	echo "expPath: ${expPath}" >> train.log
	python3 -B train.py 2>> trainerrors.log
	# Move config into generated directory
	mv "config.py" "${expPath}config_${expName}.py"
	echo "Done training ${expName}" >> train.log
	echo "Done training ${expName}"
done
echo "All experiments finished training" >> train.log
echo "All experiments finished training"
