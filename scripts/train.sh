#!/bin/bash
#SBATCH --job-name=arehalli_gulordava        # Job name (last name + short, descriptive tag-line is probably best!)
#SBATCH --output=logs/output.txt             # Standard output file (the logs folder needs to exist already!)
#SBATCH --error=logs/error.txt               # Standard error file
#SBATCH --nodes=1                            # Number of nodes (You really should keep this 1)
#SBATCH --ntasks-per-node=1                  # Number of tasks per node (also probably keep this 1)
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task (8 is probably unnecessary honestly. Should benchmark to find a sensible number...)
#SBATCH --gres=gpu:1                         # How many special resources (in this case, GPUs) do you want (in this case, 1)
#SBATCH --time=48:00:00                      # Maximum runtime (D-HH:MM:SS works, but you can also be lazy like me and )
#SBATCH --mail-type=END                      # Send email at job completion (If you'd like these kinds of updates! Honestly not sure if this works on this machine yet...)
#SBATCH --mail-user=sarehall@macalester.edu  # Email address for notifications

# From here on out, code that runs the job you want. Here I'm running a python script with some command-line args.
# Note the python "-u" flag, which unbuffers the stdout/stderr streams so if something goes wrong everything is still
# logged to the files specified above! 

python3 -u main.py --data_path ./data/ --cuda --save_path ./models/
