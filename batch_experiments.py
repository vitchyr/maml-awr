import os
import sys
import time
import subprocess
import numpy as np
import argparse
import subprocess

output_dir = "output/"

cmd_template = "python3 run_experiment.py --replay_buffer_size 500 --full_buffer_size 10000 --single_task --env walker_param --task_idx {:d} --save_dir {:s}"

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task_beg', type=int, default=0)
	parser.add_argument('--task_end', type=int, default=1)
	return parser.parse_args()
	
def run_experiments(args):
	task_beg = args.task_beg
	task_end = args.task_end
	print("Task beg: ", task_beg)
	print("Task end: ", task_end)
	
	for task_id in range(task_beg, task_end):
		curr_output_dir = os.path.join(output_dir, "task_{0:03d}".format(task_id))
		cmd = cmd_template.format(task_id, curr_output_dir)
		
		print("cmd", cmd)
		
		if not os.path.exists(curr_output_dir):
			os.mkdir(curr_output_dir)
			
		subprocess.Popen(cmd, shell=True)
	
	return

def main():
	args = get_args()
	run_experiments(args)
	return

if __name__ == '__main__':
	main()