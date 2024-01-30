import mlflow
import shutil
import os
from argparse import ArgumentParser

    
def get_run_dir(artifacts_uri):
    return artifacts_uri[18:-10]
    
def remove_run_dir(run_dir, ml_runs_path, ml_artifacts_path):
    path = os.path.join(ml_runs_path, run_dir)
    shutil.rmtree(path, ignore_errors=True)
    path = os.path.join(ml_artifacts_path, run_dir)
    shutil.rmtree(path, ignore_errors=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--experiment_id", type=int, help="experiment id", default=None)
    return parser.parse_args()

if __name__ == '__main__':

    ml_runs_path = '../logs/mlruns'
    ml_artifacts_path = '../logs/mlartifacts'

    args = parse_args()
    
    experiment_id = args.experiment_id

    if experiment_id == None:
        print('assign a experiment id')
        
    exp = mlflow.tracking.MlflowClient(tracking_uri=ml_runs_path)
        
    runs = exp.search_runs(str(experiment_id), run_view_type=2)

    _ = [remove_run_dir(get_run_dir(run.info.artifact_uri), ml_runs_path, ml_artifacts_path) for run in runs]
