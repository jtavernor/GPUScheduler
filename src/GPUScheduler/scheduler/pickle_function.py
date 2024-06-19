import sys
import os
import torch
import pathlib
from datetime import datetime

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

class PickleFunction:
    def __init__(self, log_dir, run_fn):
        self.log_dir = log_dir
        self.run_fn = run_fn

    def __call__(self, script_name, grouping=None, **args):
        torch.multiprocessing.set_sharing_strategy('file_system')
        if grouping is not None:
            mkdirp(f"{self.log_dir}/{grouping}")
            log_file = os.path.join(self.log_dir, f'{grouping}/{script_name}.log')
        else:
            mkdirp(f"{self.log_dir}")
            log_file = os.path.join(self.log_dir, f'{script_name}.log')
        sys.stdout = sys.stderr = open(log_file, 'w', buffering=1)
        start_time = datetime.now()
        print('logging begun, current time', start_time.strftime('%D %H:%M:%S'))
        self.run_fn(**args)
