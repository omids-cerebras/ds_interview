import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_data(num_servers:int=50, num_days:int=7, seed:int=42):
    """
    Generates time-series synthetic data for servers in a data center.
    Power consumption is correlated with x/y position, job type, and time patterns.
    Includes some less important covariates.
    """
    np.random.seed(seed)

    base_start_time = pd.Timestamp('2024-01-01 00:00:00')
    time_deltas = [pd.Timedelta(hours=h) for h in range(num_days * 24)]

    all_server_data = []

    # Static properties for each server
    server_static_properties = []
    for i in range(num_servers):
        server_static_properties.append({
            'server_id': f"server_{i+1:03d}",
            'x_position': np.random.rand() * 100,
            'y_position': np.random.rand() * 100,
            'base_server_age_days': np.random.randint(30, 1800),
            'initial_maintenance_days_ago': np.random.randint(0, 365),
            'firmware_version': np.random.choice(['v1.0.1', 'v1.2.3', 'v2.0.0_beta', 'v2.1.5', 'v2.2.0'], p=[0.2,0.3,0.1,0.25,0.15])
        })

    job_types_categories = ['CPU_intensive', 'IO_intensive', 'Memory_intensive', 'Idle', 'Mixed_load']

    for props in server_static_properties:
        current_job_type = np.random.choice(job_types_categories, p=[0.25, 0.2, 0.15, 0.1, 0.3])
        for h_idx, delta in enumerate(time_deltas):
            timestamp = base_start_time + delta

            # Job type might change, e.g., every 6 hours or daily for some servers
            if h_idx > 0 and h_idx % np.random.choice([6, 12, 24]) == 0: # Job change at random intervals
                 current_job_type = np.random.choice(job_types_categories, p=[0.1, 0.1, 0.1, 0.4, 0.3]) # Higher chance of idle/mixed later

            # Time-dependent features
            hour_of_day = timestamp.hour
            day_of_week = timestamp.dayofweek # Monday=0, Sunday=6

            # Simulate some dynamic "less important" covariates
            rack_temperature_celsius = 18.0 + 5 * np.sin(2 * np.pi * hour_of_day / 24) + \
                                     2 * np.sin(2 * np.pi * day_of_week / 7) + \
                                     np.random.normal(0, 1.5) # Daily and weekly temp cycle + noise
            rack_temperature_celsius = np.clip(rack_temperature_celsius, 15, 35)


            # Base power consumption
            base_power = 50  # Watts

            # Job type effect
            job_effect = 0
            if current_job_type == 'CPU_intensive':
                job_effect = 150 + np.random.normal(0, 15)
            elif current_job_type == 'IO_intensive':
                job_effect = 70 + np.random.normal(0, 10)
            elif current_job_type == 'Memory_intensive':
                job_effect = 100 + np.random.normal(0, 12)
            elif current_job_type == 'Mixed_load':
                job_effect = 120 + np.random.normal(0, 10)
            else: # Idle
                job_effect = np.random.normal(0, 5)

            # Positional effect (remains static for the server)
            positional_effect = (props['x_position'] / 100) * 30 + (props['y_position'] / 100) * 20

            # Interaction effect
            interaction_effect = 0
            if current_job_type == 'CPU_intensive' and props['x_position'] > 70:
                interaction_effect = 25

            # Time-of-day effect (e.g. higher activity during business hours)
            time_effect = 0
            if 9 <= hour_of_day <= 17 and day_of_week < 5: # Business hours on weekdays
                time_effect = 20 + np.random.normal(0,5)
            elif hour_of_day < 6 or hour_of_day > 22 : # Night time
                time_effect = -10 + np.random.normal(0,3) # Lower consumption


            current_power = base_power + job_effect + positional_effect + interaction_effect + time_effect + np.random.normal(0, 8)
            current_power = np.maximum(current_power, 10) # Min power

            all_server_data.append({
                'server_id': props['server_id'],
                'timestamp': timestamp,
                'x_position': props['x_position'],
                'y_position': props['y_position'],
                'job_type': current_job_type,
                'server_age_days': props['base_server_age_days'] + (timestamp - base_start_time).days, # Age increases over time
                'last_maintenance_days_ago': props['initial_maintenance_days_ago'] + (timestamp - base_start_time).days, # Simplified: assuming no new maintenance
                'firmware_version': props['firmware_version'], # Assuming firmware doesn't change frequently in this scope
                'rack_temperature_celsius': rack_temperature_celsius,
                'power_consumption_kw': current_power / 1000
            })

    df = pd.DataFrame(all_server_data)

    # Simulate some missing values (spread across the dataframe)
    total_rows = len(df)
    for col in ['x_position', 'job_type', 'server_age_days', 'firmware_version', 'rack_temperature_celsius']:
        # x_position is static, so if it's missing for one server, it's missing for all its timestamps
        if col == 'x_position': # Should be handled carefully if static props are NaN'd
             pass # Let's assume static props are complete for simplicity here
        else:
            nan_indices = np.random.choice(df.index, size=int(total_rows * 0.01), replace=False)
            df.loc[nan_indices, col] = np.nan

    logger.info("--- Time-Series Data Generation Complete ---")
    logger.info(f"Generated {len(df)} records ({num_servers} servers, {num_days*24} timesteps each).")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample data:\n{df.head()}\n")
    return df


def preprocess_data(dataset):
    """
    Preprocesses the time-series server data.
    Extracts time features.
    """

    # Feature Engineering: Extract time components *before* dropping columns
    dataset['hour_of_day'] = dataset['timestamp'].dt.hour
    dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    dataset['day_of_month'] = dataset['timestamp'].dt.day
    dataset['month'] = dataset['timestamp'].dt.month
    # df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int) # Requires pandas >= 1.1

    # Define features (X) and target (y) *after* adding time features
    # server_id and original timestamp might be dropped or handled differently depending on model
    X = dataset.drop(['power_consumption_kw', 'server_id', 'timestamp'], axis=1)
    y = dataset['power_consumption_kw']

    numerical_features = ['x_position', 'y_position', 'server_age_days',
                          'last_maintenance_days_ago', 'rack_temperature_celsius',
                          'hour_of_day', 'day_of_week', 'day_of_month', 'month'] # 'week_of_year'
    categorical_features = ['job_type', 'firmware_version']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # e.g. if we decide to keep server_id (encoded) for some reason
    )

    # Using random split for now. For true time-series forecasting, a chronological split is better.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    logger.info("Missing values before preprocessing (training set sample):")
    logger.info(X_train.isnull().sum().head())

    logger.info("Preprocessing setup complete.\n")
    # Return the original dataframe WITH the new time features for visualization/interpretation
    return dataset, X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features


def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor):
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True,
                                           n_jobs=-1, max_depth=15, min_samples_split=10, min_samples_leaf=5)) # Adjusted params slightly
    ])
    logger.info("Training the RandomForestRegressor model...")
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    if hasattr(model.named_steps['regressor'], 'oob_score_') and model.named_steps['regressor'].oob_score_:
         logger.info(f"Out-of-Bag R^2 score: {model.named_steps['regressor'].oob_score_:.4f}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\nModel Performance on Test Set:")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R-squared (R2) Score: {r2:.4f}\n")
    return model, y_pred

import matplotlib.pyplot as plt
import seaborn as sns

def interpret_model(model, df_with_time_features, numerical_features, categorical_features_original):
    preprocessor_fitted = model.named_steps['preprocessor']
    onehot_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']

    # Numerical features names are direct from the input list
    num_feat_names = numerical_features

    # Categorical features names come from one-hot encoder
    cat_feat_names_onehot = list(onehot_encoder.get_feature_names_out(categorical_features_original))

    all_feature_names_transformed = num_feat_names + cat_feat_names_onehot

    importances = model.named_steps['regressor'].feature_importances_

    feature_importance_df = pd.DataFrame({'feature': all_feature_names_transformed, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    logger.info("Feature Importances:")
    logger.info(feature_importance_df.head(20)) # Show more features

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20)) # Top 20
    plt.title('Top Feature Importances (Time-Series Model)')
    plt.tight_layout()
    plt.savefig("feature_importances_timeseries.png")
    logger.info("\nFeature importance plot saved as feature_importances_timeseries.png")

    # Correlation Matrix (on original numerical + engineered time features)
    logger.info("\nCorrelation Matrix (Numerical & Time Features vs Target):")
    # Use the dataframe that already has the time features
    cols_for_corr = numerical_features + ['power_consumption_kw']
    df_for_corr = df_with_time_features[cols_for_corr].copy()
    df_for_corr.dropna(inplace=True)

    if not df_for_corr.empty and len(df_for_corr.columns) > 1:
        correlation_matrix = df_for_corr.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
        plt.title('Correlation Matrix (Numerical & Time Features and Target)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("correlation_matrix_timeseries.png")
        logger.info("\nCorrelation matrix plot saved as correlation_matrix_timeseries.png")
    else:
        logger.info("Not enough data for correlation matrix.")

    # Plot power consumption over time for a sample server
    # Use the dataframe that already has the time features
    sample_server_id = df_with_time_features['server_id'].unique()[0]
    sample_server_data = df_with_time_features[df_with_time_features['server_id'] == sample_server_id].copy()

    plt.figure(figsize=(15, 6))
    plt.plot(sample_server_data['timestamp'], sample_server_data['power_consumption_kw'], marker='.', linestyle='-')
    plt.title(f'Power Consumption Over Time for {sample_server_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Power Consumption (kW)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("power_over_time_sample_server.png")
    logger.info(f"\nPower over time plot for {sample_server_id} saved as power_over_time_sample_server.png")
