import mlflow
import shutil
import os

ml_runs_path = '../logs/mlruns'
ml_artifacts_path = '../logs/mlartifacts'
    
def get_run_dir(artifacts_uri):
    return artifacts_uri[18:-10]
    
def remove_run_dir(run_dir):
    path = os.path.join(ml_runs_path, run_dir)
    shutil.rmtree(path, ignore_errors=True)
    path = os.path.join(ml_artifacts_path, run_dir)
    shutil.rmtree(path, ignore_errors=True)
    
experiment_id = 422843293446608405
# experiment_id = 378651521004183178
    
exp = mlflow.tracking.MlflowClient(tracking_uri=ml_runs_path)
    
runs = exp.search_runs(str(experiment_id), run_view_type=2)

_ = [remove_run_dir(get_run_dir(run.info.artifact_uri)) for run in runs]
