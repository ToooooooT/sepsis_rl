import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tb_to_csv(log_dir, output_path, column):
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    events = event_accumulator.Scalars(column)
    x = [x.step for x in events]
    y = [x.value for x in events]
    df = pd.DataFrame({"step": x, column: y})
    df.to_csv(output_path)