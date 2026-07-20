# feature_engineering.py

import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info(f'Memory usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def main(input_path, output_path):
    """
    Main function to load data, engineer features, and save the result.
    """
    logging.info("Starting feature engineering process.")

    # Load data
    logging.info("Loading sales, calendar, and price data...")
    sales = pd.read_csv(os.path.join(input_path, 'sales_train_validation.csv'))
    calendar = pd.read_csv(os.path.join(input_path, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(input_path, 'sell_prices.csv'))

    # Melt sales data to long format
    logging.info("Melting sales data from wide to long format...")
    # Create prediction rows for 28 days
    for d in range(1914, 1914 + 28):
        col = f"d_{d}"
        sales[col] = 0
        sales[col] = sales[col].astype(np.int16)

    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales = pd.melt(sales, id_vars=id_cols, var_name='d', value_name='sales')
    
    # Merge with calendar and prices
    logging.info("Merging with calendar and prices...")
    calendar = reduce_mem_usage(calendar)
    sales = pd.merge(sales, calendar, on='d', how='left')
    prices = reduce_mem_usage(prices)
    sales = pd.merge(sales, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    sales.drop(['d'], axis=1, inplace=True) # d is no longer needed after merge

    # Label Encoding for categorical features
    logging.info("Label encoding categorical features...")
    cat_feats = id_cols[1:] + ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in cat_feats:
        if sales[col].dtype == 'object':
            sales[col] = sales[col].astype('category')
            sales[col] = sales[col].cat.codes.astype("int16")
            sales[col] -= sales[col].min()

    # Date-based features
    logging.info("Engineering date-based features...")
    sales['date'] = pd.to_datetime(sales['date'])
    sales['day'] = sales['date'].dt.day.astype(np.int8)
    sales['week'] = sales['date'].dt.week.astype(np.int8)
    sales['month'] = sales['date'].dt.month.astype(np.int8)
    sales['year'] = sales['date'].dt.year.astype(np.int16)
    sales['dayofweek'] = sales['date'].dt.dayofweek.astype(np.int8)
    sales.drop(['date', 'wm_yr_wk'], axis=1, inplace=True)

    # Lag features
    logging.info("Engineering lag features for sales...")
    lag_days = [28, 29, 30, 31, 32, 33, 34, 35] 
    for lag in lag_days:
        sales[f'sales_lag_{lag}'] = sales.groupby(id_cols)['sales'].transform(lambda x: x.shift(lag)).astype(np.float16)

    # Rolling window features
    logging.info("Engineering rolling window features...")
    for lag in [28]:
        for window in [7, 14, 28]:
            sales[f'rolling_mean_l{lag}_w{window}'] = sales.groupby(id_cols)[f'sales_lag_{lag}'].transform(lambda x: x.rolling(window).mean()).astype(np.float16)
            sales[f'rolling_std_l{lag}_w{window}'] = sales.groupby(id_cols)[f'sales_lag_{lag}'].transform(lambda x: x.rolling(window).std()).astype(np.float16)

    # Price features
    logging.info("Engineering price features...")
    sales['price_momentum'] = sales['sell_price'] / sales.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    sales['price_momentum'] = sales['price_momentum'].astype(np.float16)
    
    # Final memory reduction
    logging.info("Performing final memory reduction...")
    sales = reduce_mem_usage(sales)
    
    # Define features to use
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'year', 'month', 'week', 'day', 'dayofweek',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'price_momentum',
    ] + [col for col in sales.columns if 'lag' in col or 'rolling' in col]
    
    logging.info(f"Total features created: {len(features)}")
    
    # Filter data for prediction period
    # We only need the data from day 1914 onwards for prediction
    # Get the integer 'd' back for filtering
    sales['d_int'] = sales['d_int'] = sales['d_int'] = sales['d_int'] = sales['d_int'] = sales['d_int'] = calendar['d'].str[2:].astype(int)
    prediction_data = sales[sales['d_int'] >= 1914].copy()
    
    logging.info("Converting float16 columns to float32 for Parquet compatibility...")
    for col in prediction_data.columns:
        if prediction_data[col].dtype == 'float16':
            prediction_data[col] = prediction_data[col].astype('float32')

    # Save the processed data
    output_file = os.path.join(output_path, 'features_for_prediction.parquet')
    logging.info(f"Saving features to {output_file}")
    prediction_data.to_parquet(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For SageMaker, paths are mounted like this. For local use, they are just folder paths.
    parser.add_argument('--input-path', type=str, default='/opt/ml/processing/input/data', help='Path to raw data.')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/output/features', help='Path to save processed features.')
    args = parser.parse_args()
    
    main(args.input_path, args.output_path)