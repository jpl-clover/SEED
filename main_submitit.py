import os
import subprocess

import submitit

from main import main
from seed.utils import parse_args


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self.set_hostname_port()
        return main(None, self.args)

    def set_hostname_port(self):
        # find a common host name on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        self.args.master_addr = host_name
        self.args.dist_url = f"tcp://{host_name}:{self.args.port}"
        os.environ["MASTER_ADDR"] = host_name
        os.environ["MASTER_PORT"] = self.args.port


# Example usage:
# ==============
# python main_submitit.py\
#     --config config/your_config.yaml \
#     --ngpus_per_node 4 \
#     --nodes 2 &

if __name__ == "__main__":
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.job_dir)
    executor.update_parameters(
        name=args.slurm_job_name,
        slurm_partition="v100",
        timeout_min=args.walltime,
        nodes=args.nodes,
        gpus_per_node=1,
        tasks_per_node=args.ngpus_per_node,  # one task per GPU
    )
    trainer = Trainer(args)
    job = executor.submit(trainer)
    results = job.results()