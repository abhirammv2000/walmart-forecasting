# scripts/predict.py (Batch Transform Compatible)

import pandas as pd
import lightgbm as lgb
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SageMaker mounts models to /opt/ml/model
MODEL_PATH = '/opt/ml/model'

def main():
    logging.info("Starting batch prediction process.")
    
    # In batch transform, the input data is mounted at /opt/ml/processing/input
    # The file name will be the same as the S3 object key.
    input_dir = '/opt/ml/processing/input'
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    
    if not input_files:
        raise ValueError("No input files found for prediction.")
        
    # Assuming one feature file. If multiple, they need to be concatenated.
    feature_file = input_files[0]
    logging.info(f"Loading features from {feature_file}")
    features_df = pd.read_parquet(feature_file)

    # Define feature columns (must be the same as in training)
    features = [col for col in features_df.columns if col not in ['id', 'sales', 'd_int']]
    
    all_preds = []
    
    # Loop through each of the 28 forecast days
    for i in range(1, 29):
        day = 1913 + i
        logging.info(f"Predicting for F{i} (day {day})...")
        
        # Unpack the model.tar.gz to get the individual model file
        model_file = os.path.join(MODEL_PATH, f'lgbm_direct_f{i}.bin')
        if not os.path.exists(model_file):
            # If model isn't in root, it's in a subfolder from tar command
            # This is a common pattern to find the file if needed.
            for root, dirs, files in os.walk(MODEL_PATH):
                if f'lgbm_direct_f{i}.bin' in files:
                    model_file = os.path.join(root, f'lgbm_direct_f{i}.bin')
                    break
        
        model = lgb.Booster(model_file=model_file)
        
        # Get data for the specific day to predict
        day_df = features_df[features_df['d_int'] == day]
        
        if day_df.empty: continue
            
        preds = model.predict(day_df[features])
        
        day_preds = pd.DataFrame({'id': day_df['id'], f'F{i}': preds})
        
        if not all_preds:
            all_preds.append(day_preds)
        else:
            all_preds[0] = pd.merge(all_preds[0], day_preds, on='id', how='outer')

    logging.info("Assembling final prediction file...")
    if not all_preds:
        raise ValueError("No predictions were generated.")
        
    final_df = all_preds[0].fillna(0)
    
    # Save as a CSV with no header, as required by Batch Transform output
    # The output will be named 'features_for_prediction.parquet.out' by default
    output_dir = '/opt/ml/processing/output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    final_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, header=False)
    logging.info("Prediction file saved successfully.")

if __name__ == '__main__':
    main()