
from audio_based_classifier.baseline_03 import train_models as audio_baseline_main
from audio_based_classifier.tree_models.tree_models_03 import run_experiment as audio_tree_models_main
#from audio_based_classifier import train_models as audio_baseline_main

def main():
    print("Running early fusion pipeline")

    ftypes = {
    #"concat_hubert_text": "concat_early_fusion.csv", 
    "concat_hubert_text": "EF_hubert_text.csv", #text + hubert
    "concat_hubert_expertk_text": "EF_hubert_text_expertk.csv" #text + hubert + expertk (opensmile mfcc and egemaps)
    #"concat_hubert_expertk_text": "expertk_hubert_text_concat_early_fusion.csv" #text + hubert + expertk (opensmile mfcc and egemaps)
    }

    oversampling_methods = ["ADASYN"]

    # Call audio-based baseline
    #audio_baseline_main(ftypes, oversampling_methods, save_path="early_fusion_results.csv")

    # try with hubert + text + expertk
    data_path = "EF_hubert_text_expertk.csv"
    # try with hubert + text 
    data_path = "EF_hubert_text_expertk.csv"
    
    metrics_path = "tree_models_results.csv"

    audio_tree_models_main(data_path,metrics_path,False, model_type="lightgbm", number_of_trials=500 )
    

    # Continue with multimodal fusion logic here


if __name__ == "__main__":
    ## RUN USING THE FOLLOWING COMMAND: python -m multimodal_fusion_model.final_early_fusion                                                                                           
    main()