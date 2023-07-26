from src import ExperimentRunner
import mlflow
from src import ExperimentConfig
from src.features import knowledge, preprocessing, sequences
from src.training import models
from src import refinement

import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def _log_all_configs_to_mlflow():
    for config in [
        ExperimentConfig(),
        preprocessing.huawei.HuaweiPreprocessorConfig(),
        preprocessing.tbird.ThunderBirdPreprocessorConfig(),
        preprocessing.hdfs.HDFSPreprocessorConfig(),
        preprocessing.mimic.MimicPreprocessorConfig(),
        preprocessing.bgl.BGLPreprocessorConfig(),
        sequences.SequenceConfig(),
        models.ModelConfig(),
        models.TextualPaperModelConfig(),
        knowledge.KnowledgeConfig(),
        refinement.RefinementConfig(),
    ]:
        for config_name, config_value in vars(config).items():
            full_config_name = config.__class__.__name__ + config_name
            mlflow.log_param(full_config_name, str(config_value))

def _main() -> str:
    mlflow.set_experiment("Attention based log template selection corrected version")
    with mlflow.start_run() as run:
        _log_all_configs_to_mlflow()
        runner = ExperimentRunner(run.info.run_id)
        runner.run()
        return run.info.run_id
