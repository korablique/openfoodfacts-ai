import os
import json
from dotenv import load_dotenv

import torch
from datasets import(
    load_dataset,
    disable_caching,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import comet_ml

from spellcheck.utils import get_logger
from spellcheck.training.configs import (
    SFTDataProcessingConfig,
    ModelConfig,
    DataConfig,
    SavingConfig,
    InferenceConfig,
    TrainingDataFeatures,
    EvaluationDataFeatures,
)
from spellcheck.training.trainer import (
    Datasets,
    SFTDataProcessor,
    LoRASavingProcessor,
    TextGenerationInference,
    CometExperimentLogger,
    EvaluationProcessor,
)
from spellcheck.training.utils import CustomCometCallback
from spellcheck.evaluation.evaluator import SpellcheckEvaluator


LOGGER = get_logger("INFO")

# For debugging
load_dotenv()

# Dataset not saved in cache to save time
disable_caching()


def main():
    LOGGER.info("Start training job.")

    ######################
    # SETUP
    ######################
    LOGGER.info("Parse information from CLI using Argparser.")
    parser = HfArgumentParser([
        SFTConfig,
        # BitsAndBytesConfig,
        # LoraConfig,
        SFTDataProcessingConfig,
        TrainingDataFeatures,
        EvaluationDataFeatures,
        ModelConfig,
        DataConfig,
        SavingConfig,
        InferenceConfig,
    ])
    (
        sft_config, 
        # quantization_config, 
        # lora_config, 
        data_processing_config,
        training_data_features,
        evaluation_data_features,
        model_config, 
        data_config, 
        saving_config, 
        inference_config,
    ) = parser.parse_args_into_dataclasses()

    #NOTE: Bug with LoraConfig and HFArgumentParser (Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because the argument parser only supports one type per argument. Problem encountered in field 'init_lora_weights'.)
    # We instantiate LoraConfig "manually"
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    torch_dtype = torch.bfloat16 if sft_config.bf16 else torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    # Sagemaker environment variables: https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    OUTPUT_DIR = os.getenv("SM_MODEL_DIR")

    #Comet experiment. Will be used in the CometCallback during the training
    EXPERIMENT_KEY = os.getenv("COMET_EXPERIMENT_KEY")
    experiment = comet_ml.ExistingExperiment(previous_experiment=EXPERIMENT_KEY)

    ######################
    # LOAD DATA
    ######################
    LOGGER.info("Load datasets.")
    training_dataset = load_dataset(
        path=data_config.training_data, 
        split=data_config.train_split,
        revision=data_config.train_data_revision,
    )
    # Evaluation dataset can be absent, for pretraining for instance
    evaluation_dataset = None
    if data_config.evaluation_data:
        evaluation_dataset = load_dataset(
            path=data_config.evaluation_data,
            split=data_config.eval_split,
            revision=data_config.eval_data_revision,
        )
    datasets = Datasets(
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset
    )
    LOGGER.info(f"Training dataset: {datasets.training_dataset}")
    LOGGER.info(f"Evaluation dataset: {datasets.evaluation_dataset}")

    ######################
    # DATA PROCESSING
    ######################
    LOGGER.info("Start pre-processing datasets.")
    data_processor = SFTDataProcessor(
        data_processsing_config=data_processing_config,
    )
    processed_datasets = data_processor.process_datasets(
        datasets=datasets,
        training_data_features=training_data_features,
        evaluation_data_features=evaluation_data_features
    )
    LOGGER.info(f"Processed training dataset: {processed_datasets.training_dataset}.")
    if processed_datasets.evaluation_dataset:
        LOGGER.info(f"Processed evaluation dataset: {processed_datasets.evaluation_dataset}.")

    ######################
    # MODEL PREPARATION
    ######################
    LOGGER.info("Start preparing the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    LOGGER.info("Start preparing the model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name,
        device_map=model_config.device_map,
        torch_dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    LOGGER.info("Model prepared succesfully for training.")

    ######################
    # TRAIN
    ######################
    LOGGER.info("Start training.")
    # Since we parse config using Argument 
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_datasets.training_dataset,
        eval_dataset=processed_datasets.evaluation_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    # There is an issue when using CometML with Trainer (starts a new experiment)
    # Thus we modified the callback to track experimentation on existing experiment
    trainer.add_callback(CustomCometCallback)
    trainer.train()
    
    ######################
    # SAVING
    ######################
    LOGGER.info("Start saving.")
    saving_processor = LoRASavingProcessor(
        output_dir=OUTPUT_DIR,
        saving_config=saving_config,
    )
    saving_processor.save_trainer(trainer=trainer)
    LOGGER.info(f"Model saved at: {saving_processor.output_dir}")

    ######################
    # EVALUATION
    ######################
    # LOGGER.info("Start evaluation.")
    # inference_processor = TextGenerationInference.load_pretrained(
    #     model_dir=sft_config.output_dir,
    #     model_config=model_config,
    #     data_processor=data_processor,
    #     inference_config=inference_config,
    # )
    # evaluation_processor = EvaluationProcessor(
    #     evaluator_type=SpellcheckEvaluator,
    #     inference_processor=inference_processor,
    #     evaluation_dataset=datasets.evaluation_dataset,
    #     evaluation_features=evaluation_data_features,
    # )
    # s3_evaluation_path = os.path.join(os.getenv("S3_EVALUATION_URI"), "evaluation-" + SM_JOB_NAME)
    # metrics = evaluation_processor.evaluate(
    #     save_predictions_path=s3_evaluation_path,
    # )
    # LOGGER.info(f"Evaluation metrics: {metrics}")
    # LOGGER.info(f"Predictions saved at: {s3_evaluation_path}")

    LOGGER.info("End of the training job.")

if __name__ == "__main__":
    main()
