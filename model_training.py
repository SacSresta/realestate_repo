"""
Real Estate Price Prediction - Model Training Pipeline
This script handles data transformation, feature engineering, and ML model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor, RANSACRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn
import subprocess
import webbrowser
import time
import os
from threading import Thread


def load_and_prepare_data(filepath):
    """Load and perform initial data cleaning"""
    df = pd.read_csv(filepath)
    
    # Date transformations
    df['soldDate'] = pd.to_datetime(df['soldDate'])
    df['Year'] = df['soldDate'].dt.year
    df['Month'] = df['soldDate'].dt.month
    df['Day'] = df['soldDate'].dt.day
    
    # Calculate annual appreciation
    median_price = df.groupby(['Year', 'pType'])['soldPrice'].median().reset_index()
    median_price.sort_values(by='Year', inplace=True)
    median_price['annual_appreciation_pct'] = median_price.groupby(['pType'])['soldPrice'].pct_change()
    
    df = df.merge(median_price[['Year', 'pType', 'annual_appreciation_pct']],
                  on=['Year', 'pType'],
                  how='left')
    
    df['annual_appreciation_pct'] = df['annual_appreciation_pct'].fillna(0)
    
    # Clean up
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df.drop(columns=['soldDate'], inplace=True)
    df.drop(columns=['Address', 'state', 'address'], inplace=True, errors='ignore')
    
    return df


def create_balanced_dataset(df):
    """Create balanced dataset by property type"""
    min_size = df['pType'].value_counts().min()
    
    balanced_df = (
        df.groupby('pType', group_keys=False)
          .apply(lambda x: x.sample(n=min_size, random_state=42))
          .reset_index(drop=True)
    )
    
    print("Balanced dataset shape:", balanced_df.shape)
    print(balanced_df['pType'].value_counts())
    
    return balanced_df


def feature_engineering(df):
    """Apply feature engineering transformations"""
    # Combine amenity features
    amenity_cols = [
        'num_dining', 'num_shopping', 'num_accommodation', 'num_entertainment',
        'num_transportation', 'num_education', 'num_leisure', 'num_healthcare',
        'num_services', 'num_religious', 'num_unknown'
    ]
    df['amenity_index'] = df[amenity_cols].sum(axis=1)
    
    # Combine sentiment features
    sentiment_cols = ['compound', 'roberta_compound', 'textblob_polarity']
    df['sentiment_score'] = df[sentiment_cols].mean(axis=1)
    
    # Drop original columns
    df = df.drop(columns=amenity_cols + sentiment_cols)
    df.drop(columns='total_attractions', inplace=True, errors='ignore')
    
    return df


def preprocess_data(df, target='soldPrice', test_size=0.05):
    """Preprocess data for modeling"""
    X = df.drop(columns=target)
    y = np.log1p(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    numeric_features = X.select_dtypes(exclude=['object', 'category']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    return X_train_prepared, X_test_prepared, y_train, y_test, preprocessor


def train_model(X_train, y_train, X_test, model):
    """Train model and make predictions"""
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def evaluate_model(y_test, predictions, return_original=True):
    """Evaluate model performance on both log-transformed and original scale"""
    # Log-transformed metrics
    mse_log = mean_squared_error(y_test, predictions)
    mae_log = mean_absolute_error(y_test, predictions)
    r2_log = r2_score(y_test, predictions)
    rmse_log = np.sqrt(mse_log)
    
    if return_original:
        # Convert back to original scale
        y_test_original = np.expm1(y_test)
        predictions_original = np.expm1(predictions)
        
        # Original scale metrics
        mse_orig = mean_squared_error(y_test_original, predictions_original)
        mae_orig = mean_absolute_error(y_test_original, predictions_original)
        r2_orig = r2_score(y_test_original, predictions_original)
        rmse_orig = np.sqrt(mse_orig)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        return {
            'log': {'mse': mse_log, 'mae': mae_log, 'r2': r2_log, 'rmse': rmse_log},
            'original': {'mse': mse_orig, 'mae': mae_orig, 'r2': r2_orig, 'rmse': rmse_orig, 'mape': mape}
        }
    else:
        return {'log': {'mse': mse_log, 'mae': mae_log, 'r2': r2_log, 'rmse': rmse_log}}


def train_evaluate(df, model):
    """Train and evaluate a single model"""
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    predictions, model = train_model(X_train, y_train, X_test, model)
    metrics = evaluate_model(y_test, predictions)

    print(f"Log-transformed - MSE: {metrics['log']['mse']:.4f}, MAE: {metrics['log']['mae']:.4f}, R²: {metrics['log']['r2']:.4f}")
    print(f"Original scale - MSE: {metrics['original']['mse']:.2f}, MAE: {metrics['original']['mae']:.2f}, R²: {metrics['original']['r2']:.4f}")
    return metrics


def train_evaluate_mlflow_multiple(df, model_dict, experiment_name="Model_Training"):
    """Train and evaluate multiple models with MLflow tracking"""
    # Set up MLflow with filesystem backend (mlruns)
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"Error with experiment: {e}")
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    
    results = []

    for name, model in model_dict.items():
        print(f"\n=== Training model: {name} ===")

        # Preprocess the data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

        # Print data diagnostics
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        numeric_features = df.drop(columns='soldPrice').select_dtypes(exclude=['object', 'category']).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        print(f"Numeric features ({len(numeric_features)}): {list(numeric_features)}")
        print(f"Categorical features ({len(categorical_features)}): {list(categorical_features)}")

        # MLflow run
        with mlflow.start_run(run_name=name):
            # Model metadata
            mlflow.log_param("model_name", name)
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_numeric_features", len(numeric_features))
            mlflow.log_param("n_categorical_features", len(categorical_features))
            
            # Log model hyperparameters
            for param, value in model.get_params().items():
                try:
                    mlflow.log_param(f"model_{param}", value)
                except:
                    pass  # Skip if parameter can't be logged
            
            print(f"Model type: {type(model).__name__}")

            # Train and evaluate
            predictions, trained_model = train_model(X_train, y_train, X_test, model)
            metrics = evaluate_model(y_test, predictions)
            
            # Extract metrics
            mse_log = metrics['log']['mse']
            mae_log = metrics['log']['mae']
            r2_log = metrics['log']['r2']
            rmse_log = metrics['log']['rmse']
            
            mse_orig = metrics['original']['mse']
            mae_orig = metrics['original']['mae']
            r2_orig = metrics['original']['r2']
            rmse_orig = metrics['original']['rmse']
            mape = metrics['original']['mape']

            # Log-transformed metrics
            mlflow.log_metric("log_MSE", mse_log)
            mlflow.log_metric("log_MAE", mae_log)
            mlflow.log_metric("log_R2", r2_log)
            mlflow.log_metric("log_RMSE", rmse_log)
            
            # Original scale metrics
            mlflow.log_metric("orig_MSE", mse_orig)
            mlflow.log_metric("orig_MAE", mae_orig)
            mlflow.log_metric("orig_R2", r2_orig)
            mlflow.log_metric("orig_RMSE", rmse_orig)
            mlflow.log_metric("orig_MAPE", mape)
            
            # Create and log visualizations
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. Actual vs Predicted
                axes[0, 0].scatter(y_test, predictions, alpha=0.5, s=10)
                axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0, 0].set_xlabel('Actual (log-transformed)', fontsize=12)
                axes[0, 0].set_ylabel('Predicted (log-transformed)', fontsize=12)
                axes[0, 0].set_title(f'{name}: Actual vs Predicted', fontsize=14, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Residuals plot
                residuals = y_test - predictions
                axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=10)
                axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
                axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
                axes[0, 1].set_ylabel('Residuals', fontsize=12)
                axes[0, 1].set_title(f'{name}: Residual Plot', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 3. Residuals distribution
                axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
                axes[1, 0].set_xlabel('Residuals', fontsize=12)
                axes[1, 0].set_ylabel('Frequency', fontsize=12)
                axes[1, 0].set_title(f'{name}: Residuals Distribution', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # 4. Error distribution
                error_pct = np.abs((y_test - predictions) / y_test) * 100
                axes[1, 1].hist(error_pct, bins=50, edgecolor='black', alpha=0.7)
                axes[1, 1].set_xlabel('Absolute Percentage Error (%)', fontsize=12)
                axes[1, 1].set_ylabel('Frequency', fontsize=12)
                axes[1, 1].set_title(f'{name}: Error Distribution', fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                # Save and log the figure
                plot_path = f"mlflow_plots/{experiment_name}_{name.replace(' ', '_')}_diagnostics.png"
                os.makedirs("mlflow_plots", exist_ok=True)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(plot_path)
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")

            # Log artifacts (using 'name' parameter instead of deprecated 'artifact_path')
            mlflow.sklearn.log_model(preprocessor, name="preprocessor")
            mlflow.sklearn.log_model(trained_model, name="model")
            print("Artifacts logged to MLflow.")

            # Console report
            print(f"\n{name} Results:")
            print(f"  Log-transformed -> MSE: {mse_log:.4f}, MAE: {mae_log:.4f}, R²: {r2_log:.4f}, RMSE: {rmse_log:.4f}")
            print(f"  Original scale  -> MSE: {mse_orig:.2f}, MAE: ${mae_orig:.2f}, R²: {r2_orig:.4f}, RMSE: ${rmse_orig:.2f}, MAPE: {mape:.2f}%")
            
            # Store results
            results.append({
                'model': name,
                'log_MSE': mse_log,
                'log_MAE': mae_log,
                'log_R2': r2_log,
                'log_RMSE': rmse_log,
                'orig_MSE': mse_orig,
                'orig_MAE': mae_orig,
                'orig_R2': r2_orig,
                'orig_RMSE': rmse_orig,
                'orig_MAPE': mape
            })
    
    return pd.DataFrame(results)


def start_mlflow_ui():
    """Start MLflow UI in a background thread"""
    # Set the tracking URI before starting UI - use mlruns directory
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    def run_ui():
        try:
            # Use mlruns directory for filesystem backend
            mlruns_path = os.path.abspath("./mlruns")
            
            print(f"Starting MLflow UI with backend: {mlruns_path}")
            subprocess.Popen(
                ["mlflow", "ui", "--backend-store-uri", mlruns_path, "--port", "5000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"Could not start MLflow UI: {e}")
    
    thread = Thread(target=run_ui, daemon=True)
    thread.start()
    time.sleep(3)  # Wait for server to start
    
    print("\n" + "="*60)
    print("MLflow UI is starting...")
    print(f"Tracking URI: {tracking_uri}")
    print(f"MLruns location: {os.path.abspath('./mlruns')}")
    print("Access UI at: http://localhost:5000")
    print("="*60 + "\n")
    
    try:
        webbrowser.open('http://localhost:5000')
    except:
        print("Please open http://localhost:5000 in your browser manually")


def check_mlflow_experiments():
    """Check and display all MLflow experiments and runs"""
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    print("\n" + "="*60)
    print("MLFLOW EXPERIMENTS SUMMARY")
    print("="*60)
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    
    if not experiments:
        print("No experiments found in the mlruns directory.")
        return
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
        
        # Get runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if len(runs) > 0:
            print(f"  Total Runs: {len(runs)}")
            print(f"  Latest Runs:")
            for idx, run in runs.head(5).iterrows():
                run_name = run.get('tags.mlflow.runName', 'Unnamed')
                status = run['status']
                print(f"    - {run_name} (Status: {status})")
        else:
            print(f"  No runs found")
    
    print("\n" + "="*60)


def visualize_results(results_df, experiment_name):
    """Create comparison visualizations for all models"""
    if results_df.empty:
        return
    
    # Create two separate figures: one for log-transformed, one for original scale
    
    # Figure 1: Log-transformed metrics
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    results_sorted_log = results_df.sort_values('log_R2', ascending=False)
    
    axes1[0, 0].barh(results_sorted_log['model'], results_sorted_log['log_R2'], color='skyblue', edgecolor='black')
    axes1[0, 0].set_xlabel('R² Score (Log-transformed)', fontsize=12)
    axes1[0, 0].set_title('Model Comparison: R² Score (Log Scale)', fontsize=14, fontweight='bold')
    axes1[0, 0].grid(True, alpha=0.3, axis='x')
    
    axes1[0, 1].barh(results_sorted_log['model'], results_sorted_log['log_MSE'], color='salmon', edgecolor='black')
    axes1[0, 1].set_xlabel('MSE (Log-transformed)', fontsize=12)
    axes1[0, 1].set_title('Model Comparison: MSE (Log Scale)', fontsize=14, fontweight='bold')
    axes1[0, 1].grid(True, alpha=0.3, axis='x')
    
    axes1[1, 0].barh(results_sorted_log['model'], results_sorted_log['log_MAE'], color='lightgreen', edgecolor='black')
    axes1[1, 0].set_xlabel('MAE (Log-transformed)', fontsize=12)
    axes1[1, 0].set_title('Model Comparison: MAE (Log Scale)', fontsize=14, fontweight='bold')
    axes1[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes1[1, 1].barh(results_sorted_log['model'], results_sorted_log['log_RMSE'], color='plum', edgecolor='black')
    axes1[1, 1].set_xlabel('RMSE (Log-transformed)', fontsize=12)
    axes1[1, 1].set_title('Model Comparison: RMSE (Log Scale)', fontsize=14, fontweight='bold')
    axes1[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path1 = f"mlflow_plots/{experiment_name}_model_comparison_log.png"
    os.makedirs("mlflow_plots", exist_ok=True)
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"\n✓ Log-transformed comparison plot saved: {output_path1}")
    plt.show()
    
    # Figure 2: Original scale metrics
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    results_sorted_orig = results_df.sort_values('orig_R2', ascending=False)
    
    axes2[0, 0].barh(results_sorted_orig['model'], results_sorted_orig['orig_R2'], color='skyblue', edgecolor='black')
    axes2[0, 0].set_xlabel('R² Score (Original Scale)', fontsize=12)
    axes2[0, 0].set_title('Model Comparison: R² Score (Original $)', fontsize=14, fontweight='bold')
    axes2[0, 0].grid(True, alpha=0.3, axis='x')
    
    axes2[0, 1].barh(results_sorted_orig['model'], results_sorted_orig['orig_MAE'], color='lightgreen', edgecolor='black')
    axes2[0, 1].set_xlabel('MAE (Original Scale $)', fontsize=12)
    axes2[0, 1].set_title('Model Comparison: MAE in Dollars', fontsize=14, fontweight='bold')
    axes2[0, 1].grid(True, alpha=0.3, axis='x')
    
    axes2[1, 0].barh(results_sorted_orig['model'], results_sorted_orig['orig_RMSE'], color='plum', edgecolor='black')
    axes2[1, 0].set_xlabel('RMSE (Original Scale $)', fontsize=12)
    axes2[1, 0].set_title('Model Comparison: RMSE in Dollars', fontsize=14, fontweight='bold')
    axes2[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes2[1, 1].barh(results_sorted_orig['model'], results_sorted_orig['orig_MAPE'], color='gold', edgecolor='black')
    axes2[1, 1].set_xlabel('MAPE (%)', fontsize=12)
    axes2[1, 1].set_title('Model Comparison: Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    axes2[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path2 = f"mlflow_plots/{experiment_name}_model_comparison_original.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Original scale comparison plot saved: {output_path2}")
    plt.show()


def get_fast_models():
    """Define lightweight models"""
    return {
        'Linear Regression': LinearRegression(n_jobs=-1),
        'Lasso Regression': Lasso(alpha=0.001, random_state=42),
        'ElasticNet Regression': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
        'Huber Regressor': HuberRegressor(),
        'Stochastic Gradient Descent': SGDRegressor(max_iter=1000, random_state=42),
        'RANSAC Regressor': RANSACRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    }


def get_heavy_models():
    """Define heavyweight models with optimized hyperparameters for faster training"""
    return {
        'Random Forest': RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            max_features='sqrt',
            min_samples_split=100,
            n_jobs=-1,
            random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=20,
            max_depth=10,
            max_features='sqrt',
            min_samples_split=100,
            n_jobs=-1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=100,
            random_state=42
        ),
        'Neural Network (MLP)': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),
    }


def main():
    """Main execution pipeline"""
    # Set tracking URI globally first - use mlruns filesystem
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLflow directory: {os.path.abspath('./mlruns')}")
    
    # Start MLflow UI
    start_mlflow_ui()
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(r'E:\Realestate_Research\notebook\data\combined_realstate_data_uncleaned.csv')
    df_full = feature_engineering(df.copy())

    # For categorical encoding of suburb if needed
    if 'suburb' in df_full.columns and df_full['suburb'].dtype == 'object':
        le = LabelEncoder()
        df_full['suburb'] = le.fit_transform(df_full['suburb'])
    
    # Train selected models on full data
    heavy_models_subset = get_heavy_models()
    df_full.drop(columns='price_bin', inplace=True, errors='ignore')
    
    print("\n" + "="*60)
    print("Training Heavy Models on Full Population Dataset")
    print("="*60)
    
    results = train_evaluate_mlflow_multiple(df_full, heavy_models_subset, experiment_name="Heavy_Models_Population_1")
    
    # Display results summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(results.to_string(index=False))
    
    # Save results to CSV
    results.to_csv('model_training_results.csv', index=False)
    print("\n✓ Results saved to: model_training_results.csv")
    
    # Create comparison visualizations
    print("\nGenerating comparison visualizations...")
    visualize_results(results, "Heavy_Models_Population")
    
    # Check what's in MLflow
    check_mlflow_experiments()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("MLflow UI: http://localhost:5000")
    print("="*60)
    
    # Keep the script running to maintain MLflow UI
    print("\nPress Ctrl+C to stop the MLflow UI server...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == "__main__":
    main()
