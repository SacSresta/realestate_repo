"""
Hyperparameter Tuning for XGBoost and LightGBM
Cross-validation with multiple hyperparameter configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
import time
import os
from datetime import datetime


def load_and_prepare_data(filepath):
    """Load and perform initial data cleaning"""
    df = pd.read_csv(filepath)
    
    # Date transformations
    df['soldDate'] = pd.to_datetime(df['soldDate'])
    df['Year'] = df['soldDate'].dt.year
    df['Month'] = df['soldDate'].dt.month
    df['Day'] = df['soldDate'].dt.day
    
    # Annual appreciation rate - market factor calculated from temporal patterns
    # This represents year-over-year market appreciation by property type
    median_price = df.groupby(['Year', 'pType'])['soldPrice'].median().reset_index()
    median_price['annual_appreciation_pct'] = median_price.groupby(['pType'])['soldPrice'].pct_change()
    df = df.merge(median_price[['Year', 'pType', 'annual_appreciation_pct']], 
                  on=['Year', 'pType'], how='left')
    df['annual_appreciation_pct'] = df['annual_appreciation_pct'].fillna(0)
    
    # Clean up
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df.drop(columns=['soldDate'], inplace=True)
    df.drop(columns=['Address', 'state', 'address'], inplace=True, errors='ignore')
    
    return df


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


def preprocess_data(df, target='soldPrice'):
    """Preprocess data for modeling - returns raw data for proper CV"""
    X = df.drop(columns=target)
    y = np.log1p(df[target])
    
    return X, y


def create_preprocessor(X):
    """Create preprocessing pipeline"""
    numeric_features = X.select_dtypes(exclude=['object', 'category']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor


def custom_scorer_log_rmse(y_true, y_pred):
    """RMSE on log scale (for cross-validation)"""
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def custom_scorer_original_rmse(y_true, y_pred):
    """RMSE on original scale (back-transformed)"""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return -np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))


def custom_scorer_r2_original(y_true, y_pred):
    """R² on original scale (back-transformed)"""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return r2_score(y_true_orig, y_pred_orig)


def custom_scorer_mape(y_true, y_pred):
    """Mean Absolute Percentage Error on original scale"""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    # Avoid division by zero
    mask = y_true_orig > 0
    return -np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100


def cross_validate_model(model, X, y, cv_folds=5, model_name="Model"):
    """Perform cross-validation with proper preprocessing in each fold"""
    print(f"\n{'='*60}")
    print(f"Cross-validating: {model_name}")
    print(f"{'='*60}")
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Create pipeline with preprocessing and model
    preprocessor = create_preprocessor(X)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Define scorers
    rmse_log_scorer = make_scorer(custom_scorer_log_rmse)
    rmse_orig_scorer = make_scorer(custom_scorer_original_rmse)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_log_scorer = make_scorer(r2_score)
    r2_orig_scorer = make_scorer(custom_scorer_r2_original)
    mape_scorer = make_scorer(custom_scorer_mape)
    
    start_time = time.time()
    
    # Perform cross-validation
    print("Running cross-validation (this may take a while)...")
    rmse_log_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=rmse_log_scorer, n_jobs=-1)
    rmse_orig_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=rmse_orig_scorer, n_jobs=-1)
    mae_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=mae_scorer, n_jobs=-1)
    r2_log_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=r2_log_scorer, n_jobs=-1)
    r2_orig_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=r2_orig_scorer, n_jobs=-1)
    mape_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=mape_scorer, n_jobs=-1)
    
    elapsed_time = time.time() - start_time
    
    # Convert negative scores back to positive
    rmse_log_scores = -rmse_log_scores
    rmse_orig_scores = -rmse_orig_scores
    mae_scores = -mae_scores
    mape_scores = -mape_scores
    
    results = {
        'model_name': model_name,
        'cv_rmse_log_mean': rmse_log_scores.mean(),
        'cv_rmse_log_std': rmse_log_scores.std(),
        'cv_rmse_orig_mean': rmse_orig_scores.mean(),
        'cv_rmse_orig_std': rmse_orig_scores.std(),
        'cv_mae_mean': mae_scores.mean(),
        'cv_mae_std': mae_scores.std(),
        'cv_r2_log_mean': r2_log_scores.mean(),
        'cv_r2_log_std': r2_log_scores.std(),
        'cv_r2_orig_mean': r2_orig_scores.mean(),
        'cv_r2_orig_std': r2_orig_scores.std(),
        'cv_mape_mean': mape_scores.mean(),
        'cv_mape_std': mape_scores.std(),
        'training_time': elapsed_time
    }
    
    print(f"Log Scale  -> RMSE: {results['cv_rmse_log_mean']:.4f} (+/- {results['cv_rmse_log_std']:.4f}), R²: {results['cv_r2_log_mean']:.4f} (+/- {results['cv_r2_log_std']:.4f})")
    print(f"Orig Scale -> RMSE: ${results['cv_rmse_orig_mean']:.2f} (+/- ${results['cv_rmse_orig_std']:.2f}), R²: {results['cv_r2_orig_mean']:.4f} (+/- {results['cv_r2_orig_std']:.4f})")
    print(f"MAE:  {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
    print(f"MAPE: {results['cv_mape_mean']:.2f}% (+/- {results['cv_mape_std']:.2f}%)")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    return results


def get_feature_importance(model, feature_names, top_n=20):
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
    return None


def plot_feature_importance(importance_df, model_name):
    """Create feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importance_sorted = importance_df.sort_values('importance', ascending=True)
    ax.barh(importance_sorted['feature'], importance_sorted['importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {len(importance_df)} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def get_xgboost_configs():
    """Define multiple XGBoost hyperparameter configurations"""
    return {
        'XGB_Baseline': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ),
        'XGB_Shallow_Fast': XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ),
        'XGB_Deep_Slow': XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ),
        'XGB_Conservative': XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.1,
            n_jobs=-1,
            random_state=42
        ),
        'XGB_Aggressive': XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.12,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            gamma=0,
            n_jobs=-1,
            random_state=42
        ),
        'XGB_Regularized': XGBRegressor(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42
        ),
    }


def get_lightgbm_configs():
    """Define multiple LightGBM hyperparameter configurations"""
    return {
        'LGBM_Baseline': LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'LGBM_Shallow_Fast': LGBMRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=31,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'LGBM_Deep_Slow': LGBMRegressor(
            n_estimators=200,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=127,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'LGBM_Conservative': LGBMRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.7,
            colsample_bytree=0.7,
            num_leaves=50,
            min_child_samples=30,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'LGBM_Aggressive': LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.12,
            subsample=0.9,
            colsample_bytree=0.9,
            num_leaves=100,
            min_child_samples=10,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'LGBM_Regularized': LGBMRegressor(
            n_estimators=120,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=64,
            reg_alpha=0.5,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
    }


def run_hyperparameter_tuning(df, cv_folds=5, experiment_name="Hyperparameter_Tuning"):
    """Run cross-validation for all hyperparameter configurations"""
    
    # Set up MLflow
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Tuning")
    print(f"Experiment: {experiment_name}")
    print(f"CV Folds: {cv_folds}")
    print(f"{'='*60}")
    
    # Preprocess data - NO FITTING HERE to avoid leakage
    print("\nPreparing data...")
    X, y = preprocess_data(df)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Get all configurations
    xgb_configs = get_xgboost_configs()
    lgbm_configs = get_lightgbm_configs()
    all_configs = {**xgb_configs, **lgbm_configs}
    
    results_list = []
    
    # Train and evaluate each configuration
    for config_name, model in all_configs.items():
        with mlflow.start_run(run_name=config_name):
            # Log hyperparameters
            params = model.get_params()
            for param, value in params.items():
                try:
                    mlflow.log_param(param, value)
                except:
                    pass
            
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("data_size", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            
            # Perform cross-validation with proper preprocessing
            cv_results = cross_validate_model(model, X, y, cv_folds, config_name)
            
            # Log metrics to MLflow
            mlflow.log_metric("cv_rmse_log_mean", cv_results['cv_rmse_log_mean'])
            mlflow.log_metric("cv_rmse_log_std", cv_results['cv_rmse_log_std'])
            mlflow.log_metric("cv_rmse_orig_mean", cv_results['cv_rmse_orig_mean'])
            mlflow.log_metric("cv_rmse_orig_std", cv_results['cv_rmse_orig_std'])
            mlflow.log_metric("cv_mae_mean", cv_results['cv_mae_mean'])
            mlflow.log_metric("cv_mae_std", cv_results['cv_mae_std'])
            mlflow.log_metric("cv_r2_log_mean", cv_results['cv_r2_log_mean'])
            mlflow.log_metric("cv_r2_log_std", cv_results['cv_r2_log_std'])
            mlflow.log_metric("cv_r2_orig_mean", cv_results['cv_r2_orig_mean'])
            mlflow.log_metric("cv_r2_orig_std", cv_results['cv_r2_orig_std'])
            mlflow.log_metric("cv_mape_mean", cv_results['cv_mape_mean'])
            mlflow.log_metric("cv_mape_std", cv_results['cv_mape_std'])
            mlflow.log_metric("training_time", cv_results['training_time'])
            
            # Train model on full dataset to get feature importances
            print(f"Training on full dataset to extract feature importances...")
            preprocessor = create_preprocessor(X)
            X_processed = preprocessor.fit_transform(X)
            
            # Get feature names after preprocessing
            cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
                X.select_dtypes(include=['object', 'category']).columns
            )
            num_features = X.select_dtypes(exclude=['object', 'category']).columns
            feature_names = list(num_features) + list(cat_features)
            
            # Fit model
            model.fit(X_processed, y)
            
            # Get and log feature importance
            importance_df = get_feature_importance(model, feature_names, top_n=20)
            if importance_df is not None:
                # Create and log feature importance plot
                fig = plot_feature_importance(importance_df, config_name)
                mlflow.log_figure(fig, f"feature_importance_{config_name}.png")
                plt.close(fig)
                
                # Log feature importance as CSV
                importance_path = f"feature_importance_{config_name}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
                print(f"✓ Feature importances logged to MLflow")
            
            # Log the trained model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            results_list.append(cv_results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('cv_r2_orig_mean', ascending=False)
    
    return results_df


def visualize_results(results_df):
    """Create visualizations comparing all configurations"""
    
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    
    # Sort by MAPE (most important metric for real estate)
    results_sorted = results_df.sort_values('cv_mape_mean', ascending=True)
    
    # 1. MAPE (Most Important!) with error bars
    axes[0, 0].barh(results_sorted['model_name'], results_sorted['cv_mape_mean'], 
                    xerr=results_sorted['cv_mape_std'], color='gold', edgecolor='black')
    axes[0, 0].set_xlabel('MAPE % (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('⭐ MAPE - Lower is Better (BEST METRIC)', fontsize=14, fontweight='bold', color='darkred')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. R² Score (Log Scale) - Secondary metric
    results_r2_sorted = results_df.sort_values('cv_r2_log_mean', ascending=True)
    axes[0, 1].barh(results_r2_sorted['model_name'], results_r2_sorted['cv_r2_log_mean'], 
                    xerr=results_r2_sorted['cv_r2_log_std'], color='lightblue', edgecolor='black')
    axes[0, 1].set_xlabel('R² Score Log (Mean ± Std)', fontsize=12)
    axes[0, 1].set_title('CV R² Scores (Log Scale) - Higher is Better', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. R² Score (Original Scale) - Less reliable
    results_r2_orig_sorted = results_df.sort_values('cv_r2_orig_mean', ascending=True)
    axes[0, 2].barh(results_r2_orig_sorted['model_name'], results_r2_orig_sorted['cv_r2_orig_mean'], 
                    xerr=results_r2_orig_sorted['cv_r2_orig_std'], color='skyblue', edgecolor='black')
    axes[0, 2].set_xlabel('R² Score Original (Mean ± Std)', fontsize=12)
    axes[0, 2].set_title('CV R² Scores (Original $) - Skewed by Outliers', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. RMSE (Log Scale) with error bars
    results_rmse_log_sorted = results_df.sort_values('cv_rmse_log_mean', ascending=True)
    axes[1, 0].barh(results_rmse_log_sorted['model_name'], results_rmse_log_sorted['cv_rmse_log_mean'], 
                    xerr=results_rmse_log_sorted['cv_rmse_log_std'], color='lightsalmon', edgecolor='black')
    axes[1, 0].set_xlabel('RMSE Log (Mean ± Std)', fontsize=12)
    axes[1, 0].set_title('CV RMSE (Log Scale) - Lower is Better', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. RMSE (Original Scale) with error bars
    results_rmse_sorted = results_df.sort_values('cv_rmse_orig_mean', ascending=True)
    axes[1, 1].barh(results_rmse_sorted['model_name'], results_rmse_sorted['cv_rmse_orig_mean'], 
                    xerr=results_rmse_sorted['cv_rmse_orig_std'], color='salmon', edgecolor='black')
    axes[1, 1].set_xlabel('RMSE Original $ (Mean ± Std)', fontsize=12)
    axes[1, 1].set_title('CV RMSE (Original $) - Lower is Better', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. MAE with error bars
    results_mae_sorted = results_df.sort_values('cv_mae_mean', ascending=True)
    axes[1, 2].barh(results_mae_sorted['model_name'], results_mae_sorted['cv_mae_mean'], 
                    xerr=results_mae_sorted['cv_mae_std'], color='lightgreen', edgecolor='black')
    axes[1, 2].set_xlabel('MAE (Mean ± Std)', fontsize=12)
    axes[1, 2].set_title('CV MAE - Lower is Better', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    # 7. Training Time
    results_time_sorted = results_df.sort_values('training_time', ascending=True)
    axes[2, 0].barh(results_time_sorted['model_name'], results_time_sorted['training_time'], 
                    color='plum', edgecolor='black')
    axes[2, 0].set_xlabel('Training Time (seconds)', fontsize=12)
    axes[2, 0].set_title('Cross-Validation Training Time', fontsize=14, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3, axis='x')
    
    # 8. Metric Comparison Table
    axes[2, 1].axis('off')
    table_data = results_df[['model_name', 'cv_mape_mean', 'cv_r2_log_mean', 'training_time']].sort_values('cv_mape_mean').head(5)
    table_text = "Top 5 Models by MAPE:\n\n"
    for idx, row in table_data.iterrows():
        table_text += f"{row['model_name'][:20]}\n  MAPE: {row['cv_mape_mean']:.2f}%  R²: {row['cv_r2_log_mean']:.3f}  Time: {row['training_time']:.1f}s\n\n"
    axes[2, 1].text(0.1, 0.5, table_text, fontsize=10, family='monospace', verticalalignment='center')
    axes[2, 1].set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    # 9. Hide extra subplot
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "hyperparameter_tuning_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.show()


def main():
    """Main execution pipeline"""
    
    print("\n" + "="*60)
    print("XGBoost & LightGBM Hyperparameter Tuning with Cross-Validation")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data(r'E:\Realestate_Research\notebook\data\combined_realstate_data_uncleaned.csv')
    df = feature_engineering(df.copy())
    
    # Encode categorical variables
    if 'suburb' in df.columns and df['suburb'].dtype == 'object':
        le = LabelEncoder()
        df['suburb'] = le.fit_transform(df['suburb'])
    
    df.drop(columns='price_bin', inplace=True, errors='ignore')
    
    print(f"Dataset shape: {df.shape}")
    
    # Run hyperparameter tuning
    results = run_hyperparameter_tuning(df, cv_folds=5, experiment_name="XGB_LGBM_Hyperparameter_Tuning")
    
    # Display results
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print("\nTop 5 Configurations by R² Score:")
    print(results[['model_name', 'cv_r2_mean', 'cv_r2_std', 'cv_rmse_mean', 'cv_mae_mean', 'training_time']].head(10).to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_tuning_results_{timestamp}.csv"
    results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(results)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning Complete!")
    print("Check MLflow UI for detailed run information")
    print("="*60)


if __name__ == "__main__":
    main()
