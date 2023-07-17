import os
import sys
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json

# Local imports
from config import SciFactT5Config
from multivers.data_r import ClaimDataLoaderGenerator
from multivers.model_r import MultiVerSModel
from multivers import util

import definitions

# Add necessary paths to sys.path
sys.path.append(os.path.dirname(definitions.PROJECT_VARS.ROOT_DIR))

from T5ParEvo.src.data.data import Claim, ClaimPredictions, GoldDataset, Label
from T5ParEvo.src.linguistic.ner_abbr import Abbreviation, NEREntity
from T5ParEvo.target_system.multivers.multivers_interface import ModelPredictorMultivers, PredictionParams,ModelPredictorMultiversList
from T5ParEvo.src.paraphrase.paraphrase_claim import ClaimState, ParaphrasedAttack, ParaphrasedClaim, ParaphrasedAttackResult, TorchEntailmentPredictionModel
from T5ParEvo.src.util.logger import LoggerConfig, NeptuneConfig, LogConfigurator, NeptuneRunner, Logger, LightningLogger
from T5ParEvo.src.models.fine_tune import FineTuneHyperParams, T5FineTuner, LoggingCallback
from T5ParEvo.src.paraphrase.paraphraser import T5Paraphraser, ModelConfig
from T5ParEvo.src.data.dataset_preparation import DatasetPreparation
from T5ParEvo.src.paraphrase.paraphrase_claim import AttackStatus



# from src.data.data import Claim, ClaimPredictions, GoldDataset, Label
# from src.linguistic.ner_abbr import Abbreviation, NEREntity
# from target_system.multivers.multivers_interface import ModelPredictorMultivers, PredictionParams,ModelPredictorMultiversList
# from src.paraphrase.paraphrase_claim import ClaimState, ParaphrasedAttack, ParaphrasedClaim, ParaphrasedAttackResult, TorchEntailmentPredictionModel
# from src.util.logger import LoggerConfig, NeptuneConfig, LogConfigurator, NeptuneRunner, Logger, LightningLogger
# from src.models.fine_tune import FineTuneHyperParams, T5FineTuner, LoggingCallback
# from src.paraphrase.paraphraser import T5Paraphraser, ModelConfig
# from src.data.dataset_preparation import DatasetPreparation
# from src.paraphrase.paraphrase_claim import AttackStatus

# Load T5 model and tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Constants
TRAINING_DIRECTION : ClaimState = ClaimState.SUPPORT_MAJORITY
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAPHRASE_MODEL_CHECKPOINT_PATH_URL = '/home/qudratealahyratu/research/nlp/fact_checking/my_work/SciMedAttack/results/t5_paws_masked_claim_abstract_paws_3_epoch_2/models/model_3_epochs/'
PARAPHRASE_MODEL_TOKENIZER = 'Vamsi/T5_Paraphrase_Paws'
PARAPHRASE_CONFIG_PARAMS = {
    'max_length': 512,
    'do_sample': True,
    'top_k': 50,
    'top_p': 0.99,
    'repetition_penalty': 3.5,
    'early_stopping': True,
    'num_return_sequences': 10
}

SPLIT_SIZE = 0.2
NUM_EPOCHS = 10
DATA_DIR = Path('/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/data/multivers_res/')


def setup_logging():
    # Configuration for logging
    os.environ['NEPTUNE_API_TOKEN'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NWQwMGIyZi1mNzM5LTRiMjEtOTg2MC1mNTc4ODRiMWU2ZGYifQ=='
    log_config = LoggerConfig()
    log_configurator = LogConfigurator(log_config)
    log_configurator.configure()

    neptune_config = NeptuneConfig(project_name="ratulalahy/scifact-paraphrase-T5-evo",
                                tags=['other model attack', 'tech_term', 'mlnli'],
                                source_files=["main.py", "*.yaml", "config.py", "definition.py"])

    neptune_runner = NeptuneRunner(neptune_config)
    nep_run = neptune_runner.run()

    logger = Logger(nep_run, log_configurator)
    lightning_logger = LightningLogger(logger)

    return lightning_logger

def load_t5_model(checkpoint_path):
    # Utility function to load T5 model
    model_t5 = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model_t5 = model_t5.to(DEVICE)
    return model_t5

def main():
    iteration_counter = 0
    # Configure Logging
    lightning_logger = setup_logging()
    lightning_logger.log_hyperparams(PARAPHRASE_CONFIG_PARAMS)
    lightning_logger.experiment.log('TRAINING_DIRECTION', TRAINING_DIRECTION)

    # Loading T5 model and tokenizer
    model_t5 = load_t5_model(PARAPHRASE_MODEL_CHECKPOINT_PATH_URL)
    tokenizer_t5 = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_TOKENIZER)

    # Configuration
    cfg = SciFactT5Config()
    params = PredictionParams(
        checkpoint_path= "/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/checkpoints/scifact.ckpt",
        output_file= None,#"prediction/pred_opt_scifact.jsonl",
        batch_size=5,
        device=0,
        num_workers=4,
        no_nei=False,
        force_rationale=False,
        debug=False,
        corpus_file = cfg.target_dataset.loc_target_dataset_corpus
    )
    corpus_file = cfg.target_dataset.loc_target_dataset_corpus#cfg.target_dataset.loc_target_dataset_test#"/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/data/scifact/corpus.jsonl"
       

    ## Preparing data for the experiment

    gold_claims = []
    claims_path = cfg.target_dataset.loc_target_dataset_test#'/home/qudratealahyratu/research/nlp/fact_checking/my_work/multivers/data/scifact/claims_test_retrived.jsonl'
    with open(claims_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            claim = Claim(id = data['id'], claim = data['claim'], cited_docs = data['doc_ids'], evidence = {},release = None)
            gold_claims.append(claim)

    #get unique claims
    unique_gold_claims = Claim.get_unique_claims(gold_claims)

    ## Predict original Claims
    # Loading unique claims and preparing prediction model
    unique_gold_claims = Claim.get_unique_claims(gold_claims)
    dataloader_generator = ClaimDataLoaderGenerator(params, unique_gold_claims[0], corpus_file)
    dataloader = dataloader_generator.get_dataloader_by_single_claim()
    # prediction_model = ModelPredictorMultivers(params, dataloader, corpus_file)
    prediction_model = ModelPredictorMultivers(params, unique_gold_claims[0])
    original_claim_predictions_raw = prediction_model.predict(unique_gold_claims[0])


    # ## Predicting for unique claims
    all_original_claim_predictions : List[ClaimPredictions]= []
    for cur_uniq_claim in tqdm(unique_gold_claims[:], desc="Predicting for unique claims"):
        original_claim_prediction = prediction_model.predict(cur_uniq_claim)
        all_original_claim_predictions.append(original_claim_prediction)

    ## LOAD NERS and entailment model
    with open('/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/data/meta/merged_abbreviations.pkl', 'rb') as f:
        MERGED_ABBREVIATIONS= pickle.load(f)

    with open('/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/data/meta/merged_entities.pkl', 'rb') as f:
        MERGED_ENTITIES = pickle.load(f)    

    entailment_model = TorchEntailmentPredictionModel(model_path= 'pytorch/fairseq', model_name = 'roberta.large.mnli', device= DEVICE)
    

    ## PARAPHRASER
    # Load T5 model and tokenizer
    model_t5 = load_t5_model(PARAPHRASE_MODEL_CHECKPOINT_PATH_URL)
    tokenizer_t5 = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_TOKENIZER)

    # Initialize paraphrase model and paraphrase attack
    paraphrase_config = ModelConfig(**PARAPHRASE_CONFIG_PARAMS)
    paraphrase_model = T5Paraphraser(model_t5, tokenizer_t5, paraphrase_config)

    # Initialize paraphrase attack
    paraphrase_attack_model = ParaphrasedAttack(paraphrase_model, prediction_model,entailment_model ,list_ners = MERGED_ENTITIES)

    while True:
        # FIRST ATTACK! 
        all_paraphrased_attacks : List[ParaphrasedClaim] = [] # okay! `all_paraphrased_attacks` not a list of attacks. Actually list of all paraphrased claims with predicts
        for cur_original_claim_pred in tqdm(all_original_claim_predictions[:], desc="Paraphrasing claims"):
            paraphrased_attack = paraphrase_attack_model.attack(iteration = iteration_counter, 
                                                          original_claim= cur_original_claim_pred.gold, 
                                                          original_prediction = cur_original_claim_pred, 
                                                          predict_if_pass_filter=False)
        all_paraphrased_attacks.append(paraphrased_attack)

        #read pickle file
        # with open('/home/qudratealahyratu/research/nlp/fact_checking/my_work/T5ParEvo/notebooks/all_attacks.pkl', 'rb') as f:
        #     all_attack_results = pickle.load(f) 
        # for cur_res in all_attack_results:    
        #     cur_res.determine_attack_status()
        #     cur_res.training_direction = TRAINING_DIRECTION
        # Post processing the attack results. like filtering, majority and so on.
        all_attack_results : List[ParaphrasedAttackResult] = []
        for cur_claims_attack in all_paraphrased_attacks:
            for cur_attack in cur_claims_attack:
                paraphrase_attack_model.calculate_and_set_claim_states(cur_attack)
                cur_res = ParaphrasedAttackResult(cur_attack)
                cur_res.determine_attack_status()
                cur_res.training_direction = TRAINING_DIRECTION
                all_attack_results.append(cur_res)

        # save paraphrased claims with predicts and log
        f_n_all_paraphrased = f"all_paraphrased_{TRAINING_DIRECTION.value}_{iteration_counter}.pkl" 
        f_n_all_paraphrased = DATA_DIR /f_n_all_paraphrased
        with open(f_n_all_paraphrased, 'wb') as f:
            pickle.dump(all_attack_results, f)
        lightning_logger.log_metrics({'paraphrase_count': len(all_attack_results)}, step=iteration_counter)   
          
        # Reporting attack and filter data that will be used to fine tune

        
        unique_ids_attacks = set()
        attacks_to_be_used_for_training = []
        attacks_successful = []
        for cur_atk in all_attack_results:
            if cur_atk.attack_status == AttackStatus.SUCCESSFUL:
                attacks_successful.append(cur_atk)
                unique_ids_attacks.add(cur_atk.attack.original_claim.id)
                if cur_atk.training_direction == TRAINING_DIRECTION:
                    attacks_to_be_used_for_training.append(cur_atk)  
                    
        # save paraphrased claims with predicts and log
        f_n_all_attacks = f"all_attacks_{TRAINING_DIRECTION.value}_{iteration_counter}.pkl" 
        f_n_all_attacks = DATA_DIR /f_n_all_attacks
        with open(f_n_all_attacks, 'wb') as f:
            pickle.dump(attacks_successful, f)

        lightning_logger.log_metrics({'attack_count': len(attacks_successful)}, step=iteration_counter)  
        lightning_logger.log_metrics({'unique_attack_count': len(unique_ids_attacks)}, step=iteration_counter)  
        lightning_logger.log_metrics({'finetune_sample_count': len(attacks_to_be_used_for_training)}, step=iteration_counter)  
        
        #save attacks claims with predicts and log

        # Processing data for tuning       
        fine_tuning_data = {
        "org_claim": [cur_atk.attack.original_claim.claim for cur_atk in attacks_to_be_used_for_training],
        "gen_claim": [cur_atk.attack.paraphrased_claim.claim for cur_atk in attacks_to_be_used_for_training]
        }
        df_fine_tuning_dataset = pd.DataFrame(fine_tuning_data)

        prep = DatasetPreparation(df_fine_tuning_dataset, SPLIT_SIZE)
        df_tune_train, df_tune_val = prep.split_and_reset_index()


        # Getting ready to fine-tune
        fineTuneHyperParam = FineTuneHyperParams(model_name_path = PARAPHRASE_MODEL_CHECKPOINT_PATH_URL, 
                                                 tokenizer_name_path = PARAPHRASE_MODEL_TOKENIZER,
                                                     num_train_epochs = NUM_EPOCHS, 
                                                     df_train = df_tune_train, 
                                                     df_val = df_tune_val, 
                                                     df_train_val = df_fine_tuning_dataset)

        # Setup early stopping
        early_stop_callback = EarlyStopping(
           monitor='val_loss',
           patience=3,
           verbose=False,
           mode='min'
        )

        # Training model
        fineTuneHyperParam = FineTuneHyperParams(model_name_path = PARAPHRASE_MODEL_CHECKPOINT_PATH_URL, 
                                                tokenizer_name_path = PARAPHRASE_MODEL_TOKENIZER,
                                                num_train_epochs = NUM_EPOCHS, 
                                                df_train = df_tune_train, 
                                                df_val = df_tune_val, 
                                                df_train_val = df_fine_tuning_dataset)

        model_t5_fine_tuned = T5FineTuner(fineTuneHyperParam)
        trainer_model_t5_fine_tune = pl.Trainer(callbacks=[early_stop_callback, 
                                                            fineTuneHyperParam.get_checkpoint_callback(), 
                                                            LoggingCallback()], 
                                                logger=lightning_logger, 
                                                **fineTuneHyperParam.get_train_params())
        trainer_model_t5_fine_tune.fit(model_t5_fine_tuned)

        ## re-initialize attack model with fine_tuned model
        paraphrase_config = ModelConfig(**PARAPHRASE_CONFIG_PARAMS)
        paraphrase_model = T5Paraphraser(model_t5_fine_tuned.model, tokenizer_t5, paraphrase_config)

        # Initialize paraphrase attack
        paraphrase_attack_model = ParaphrasedAttack(paraphrase_model, prediction_model,entailment_model ,list_ners = MERGED_ENTITIES)
        iteration_counter += 1
        
        print(f'after iteration {iteration_counter} :: attack_count         : {len(attacks_successful)}')  
        print(f'after iteration {iteration_counter} :: unique_attack_count  : {len(unique_ids_attacks)}')
        print(f'after iteration {iteration_counter} :: finetune_sample_count: {len(attacks_to_be_used_for_training)}')

if __name__ == '__main__':
    main()
