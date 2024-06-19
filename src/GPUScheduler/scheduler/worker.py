import multiprocessing as mp
import traceback
import os
import sys
from datetime import datetime

class Worker(mp.Process):
    def __init__(self, job_queue, results_queue, gpu, run_fn):
        super(Worker, self).__init__()
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.gpu = gpu
        self.run_script = run_fn
        # Set environment variables for running pytorch scripts only on current GPU
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.gpu}'
        print(f'Worker initialised on GPU {self.gpu}')

    def run(self):
        print(f'Worker ({os.getpid()}) ready for jobs on GPU {self.gpu}')
        # Once shutdown worker is read on the job queue the worker will be killed
        for (script_name, allow_crash, args) in iter(self.job_queue.get, 'shutdown worker'):
            print(f'Worker ({os.getpid()}) running job {script_name}')
            start_time = datetime.now()
            results_item = script_name
            try:
                self.run_script(script_name, **args)
            except:
                the_type, the_value, the_traceback = sys.exc_info()
                print(f'ERROR -- {script_name} crashed continuing with other processes...')
                print('crash information')
                print(the_type)
                print(the_value)
                traceback.print_tb(the_traceback, file=sys.stdout)
                results_item = f'BAD_{results_item}'
                if not allow_crash:
                    # Can't pickle traceback so don't try to store this
                    results_item = (results_item, (the_type, the_value))
                temp = sys.stdout
                sys.stdout = sys.__stdout__
                print(f'ERROR -- {script_name} crashed (see log for info) continuing with other processes...')
                sys.stdout = temp
            finally:
                print('Experiment start time', start_time.strftime('%D %H:%M:%S'))
                print('Experiment end time', datetime.now().strftime('%D %H:%M:%S'))
                sys.stdout = sys.__stdout__
                # After finishing a job register it as finished
                self.results_queue.put(results_item)
