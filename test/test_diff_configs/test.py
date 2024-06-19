import os
import pathlib
import torch
import multiprocessing as mp

file_path = os.path.realpath(__file__).replace('/test.py', '')
def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

from GPUScheduler import Scheduler

def run_experiments(allow_crash=False):
    # Do this immediately to ensure that it isn't reset in other methods
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')

    # Define the config for use in the code
    log_dir = os.path.join(file_path, 'logs')
    mkdirp(log_dir)
    scheduler = Scheduler(num_scripts_per_gpu=2, log_dir=log_dir, additional_config={})
    print('creating workers')
    scheduler.create_workers(run_fn=ProcessRun())
    jobs = {f'script_name_{i}': {'model_config': {'num_layers:', i}, 'script id': i} for i in range(9)}
    jobs['script_name_4']['random_arg_on_run_4'] = 'random arg'
    print('launching jobs')
    scheduler.launch_jobs(jobs, allow_crash)
    print('closing processes')
    scheduler.close_processes()

import time
class ProcessRun:
    def __call__(self, **args):
        print('job begun with args:', args)
        print('num gpus available to script:', torch.cuda.device_count())
        a = torch.randn((500,500), device='cuda')
        print('created cuda variable with mean:', a.mean())
        print('sleeping for 10 seconds')
        time.sleep(10)
        print('job finishing')

if __name__ == '__main__':
    run_experiments()