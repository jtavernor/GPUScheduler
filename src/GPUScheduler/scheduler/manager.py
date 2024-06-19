import torch
import multiprocessing as mp
import sys
from .worker import Worker
from .pickle_function import PickleFunction

# Workers have an expensive start up and shutdown cost
# so what we do is launch the maximum amount of workers across GPUs and wait for jobs
# results queue will be used to ensure that all jobs have been completed 
class WorkerManager:
    def __init__(self, num_scripts_per_gpu, log_dir):
        self.num_scripts_per_gpu = num_scripts_per_gpu # How many jobs can be run simultaneously on a GPU
        self.num_gpus = torch.cuda.device_count()
        self.num_processes = self.num_gpus * self.num_scripts_per_gpu
        self.job_queue = mp.Queue()
        self.results_queue = mp.Queue()
        self.log_dir = log_dir
        # self.available_gpus = mp.Queue()
        # Will add e.g. for 3 GPUs [0,0,0,1,1,1,2,2,2] -- subprocesses will be given items in this list assigning them a GPU
        self.available_gpus = [gpu for gpu in range(self.num_gpus) for _ in range(self.num_scripts_per_gpu)]
        self.processes = []

    def create_workers(self, run_fn):
        for gpu in self.available_gpus:
            process = Worker(self.job_queue, self.results_queue, gpu, PickleFunction(self.log_dir, run_fn))
            self.processes.append(process)
            process.start()

    def graceful_shutdown(self):
        # First clear the queue to prevent any unlaunched jobs from launching
        while not self.job_queue.empty():
            self.job_queue.get()
        # Passing 'shutdown worker' on the job queue will nicely shutdown a worker allowing it to finish it's current job
        # unless there is a crash this shutdown should always be used 
        for _ in range(self.num_processes):
            self.job_queue.put('shutdown worker')

    def launch_jobs(self, job_args, allow_crash=True):
        for script_name in job_args:
            # Tuple of script_name and the arguments for that script
            self.job_queue.put((script_name, allow_crash, job_args[script_name]))

    def wait_for_jobs(self, job_names):
        finished_jobs = []
        failed_jobs = []
        print('Waiting for jobs to finish:', job_names)
        while sorted(finished_jobs) != sorted(job_names):
            finished_job = self.results_queue.get(block=True, timeout=None)
            if type(finished_job) != str:
                print('Found a crashed job where allow_crash=False')
                script_name, exception_info = finished_job
                print(f'Crash information for {script_name}:')
                print(exception_info[0])
                print(exception_info[1])
                print('see script log for traceback information')
                print('Shutting down')
                self.graceful_shutdown()
                exit()
            else:
                if finished_job.startswith('BAD_'):
                    finished_job = finished_job.replace('BAD_', '')
                    failed_jobs.append(finished_job)
                finished_jobs.append(finished_job)

            print('Finished job(s):', finished_job, '(', finished_jobs, ')')
            print('Remaining jobs', set(job_names) - set(finished_jobs))
            print('Crashed jobs', failed_jobs)