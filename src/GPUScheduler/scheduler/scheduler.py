from .manager import WorkerManager
import traceback

class Scheduler:
    def __init__(self, num_scripts_per_gpu, log_dir, additional_config):
        self.num_scripts_per_gpu = num_scripts_per_gpu
        self.log_dir = log_dir
        self.additional_config = additional_config
        self.manager = WorkerManager(self.num_scripts_per_gpu, self.log_dir)

    def create_workers(self, run_fn):
        try:
            self.manager.create_workers(run_fn)
        except Exception as e:
            print('Error: Attempting to close processes..')
            traceback.print_tb(e)
            self.close_processes

    def launch_jobs(self, jobs, allow_crash=False):
        try:
            self.manager.launch_jobs(jobs, allow_crash=allow_crash)
            self.manager.wait_for_jobs(jobs.keys())
        except Exception as e:
            print('Error: Attempting to close processes..')
            traceback.print_tb(e)
            self.close_processes

    def close_processes(self):
        self.manager.graceful_shutdown()
        for process in self.manager.processes:
            process.join()
        self.manager.job_queue.close()
        self.manager.results_queue.close()
        self.manager.job_queue.join_thread()
        self.manager.results_queue.join_thread()
