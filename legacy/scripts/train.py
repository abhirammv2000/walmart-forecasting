# scripts/train.py

import pandas as pd
import numpy as np
import os
import argparse
import logging
import lightgbm as lgb
import gc
from sklearn.preprocessing import LabelEncoder

# --- Global Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# M5 competition training data ends on day 1913
TRAIN_END_DAY = 1913
# We will use the last 28 days of the training data as a validation set
VALID_END_DAY = TRAIN_END_DAY
VALID_START_DAY = VALID_END_DAY - 28


def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def feature_engineering(input_path):
    """
    Loads data and performs all feature engineering. This must be IDENTICAL to the prediction feature engineering.
    """
    logging.info("Loading raw data...")
    sales = pd.read_csv(os.path.join(input_path, 'sales_train_validation.csv'))
    calendar = pd.read_csv(os.path.join(input_path, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(input_path, 'sell_prices.csv'))

    logging.info("Melting sales data from wide to long format...")
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales = pd.melt(sales, id_vars=id_cols, var_name='d', value_name='sales')

    # --- START OF FIX ---
    # Create d_int in sales for initial filtering only
    sales['d_int'] = sales['d'].str[2:].astype(int)
    sales = sales[sales['d_int'] <= TRAIN_END_DAY]
    # Drop the temporary d_int column from sales before merging to avoid a column name clash
    sales.drop(columns=['d_int'], inplace=True)
    # --- END OF FIX ---

    logging.info("Merging with calendar and prices...")
    calendar['d_int'] = calendar['d'].str[2:].astype(int)
    # Now, the d_int column will be added cleanly from the calendar dataframe
    sales = pd.merge(sales, calendar[['d', 'wm_yr_wk', 'date', 'd_int', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']], on='d', how='left')
    sales = pd.merge(sales, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    sales.drop(['d'], axis=1, inplace=True)
    sales = reduce_mem_usage(sales)

    logging.info("Encoding categorical features...")
    cat_feats = id_cols[1:] + ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in cat_feats:
        if sales[col].dtype == 'object':
            sales[col] = sales[col].astype('category')
            sales[col] = sales[col].cat.codes.astype("int16")
            sales[col] -= sales[col].min()

    logging.info("Engineering date-based features...")
    sales['date'] = pd.to_datetime(sales['date'])
    sales['day'] = sales['date'].dt.day.astype(np.int8)
    sales['week'] = sales['date'].dt.isocalendar().week.astype(np.int8) # Corrected deprecated call
    sales['month'] = sales['date'].dt.month.astype(np.int8)
    sales['year'] = sales['date'].dt.year.astype(np.int16)
    sales['dayofweek'] = sales['date'].dt.dayofweek.astype(np.int8)
    sales.drop(['date', 'wm_yr_wk'], axis=1, inplace=True)

    logging.info("Engineering lag and rolling window features...")
    lag_days = [28, 29, 30, 31, 32, 33, 34, 35]
    for lag in lag_days:
        sales[f'sales_lag_{lag}'] = sales.groupby(id_cols)['sales'].transform(lambda x: x.shift(lag)).astype(np.float16)

    for lag in [28]:
        for window in [7, 14, 28]:
            sales[f'rolling_mean_l{lag}_w{window}'] = sales.groupby(id_cols)[f'sales_lag_{lag}'].transform(lambda x: x.rolling(window).mean()).astype(np.float16)
            sales[f'rolling_std_l{lag}_w{window}'] = sales.groupby(id_cols)[f'sales_lag_{lag}'].transform(lambda x: x.rolling(window).std()).astype(np.float16)
    
    logging.info("Engineering price features...")
    sales['price_momentum'] = sales['sell_price'] / sales.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    sales['price_momentum'] = sales['price_momentum'].astype(np.float16)

    return sales

def main(input_path, model_output_path):
    """
    Main function to run the training process.
    """
    df = feature_engineering(input_path)

    # Define features to use
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'year', 'month', 'week', 'day', 'dayofweek',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'price_momentum'
    ] + [col for col in df.columns if 'lag' in col or 'rolling' in col]

    categorical_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    logging.info(f"Using {len(features)} features for training.")

    # LightGBM parameters (tuned for performance and speed)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.6,
        'feature_fraction': 0.4,
        'learning_rate': 0.03,
        'num_leaves': 2**8-1,
        'min_data_in_leaf': 2**8-1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 2000,
        'early_stopping_round': 100,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    # --- Direct Training Loop ---
    for i in range(1, 29):
        logging.info(f"--- Training Model for F{i} ---")
        
        # The target variable is the sales 'i' days in the future
        df['target'] = df.groupby(['id'])['sales'].transform(lambda x: x.shift(-i))
        
        # Remove rows where the target is NaN (i.e., the last few days of the dataset)
        train_df = df.dropna(subset=['target'])
        
        # Create train and validation sets
        X_train = train_df[train_df['d_int'] < VALID_START_DAY][features]
        y_train = train_df[train_df['d_int'] < VALID_START_DAY]['target']
        
        X_valid = train_df[train_df['d_int'] >= VALID_START_DAY][features]
        y_valid = train_df[train_df['d_int'] >= VALID_START_DAY]['target']

        logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_valid.shape}")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features, free_raw_data=False)

        # Train the model
        estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.log_evaluation(period=100)],
        )

        # Save the model
        model_filename = os.path.join(model_output_path, f'lgbm_direct_f{i}.bin')
        estimator.save_model(model_filename)
        logging.info(f"Saved model for F{i} to {model_filename}")

        # Clean up memory
        del X_train, y_train, X_valid, y_valid, train_data, valid_data, estimator
        gc.collect()

    logging.info("All 28 models have been trained successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define paths
    parser.add_argument('--input-path', type=str, default='/opt/ml/input/data/data', help='Path to raw data.')
    parser.add_argument('--model-output-path', type=str, default='/opt/ml/model', help='Path to save trained models.')

    # --- THIS IS THE FIX ---
    # Use parse_known_args() to ignore extra arguments passed by SageMaker
    args, unknown = parser.parse_known_args()

    # Create model output directory if it doesn't exist
    # Note: SageMaker creates the main output dir, but let's be safe.
    if not os.path.exists(args.model_output_path):
        os.makedirs(args.model_output_path)

    main(args.input_path, args.model_output_path)
