import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AirQualityPredictor:
    def __init__(self, forecast_horizon=5):
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        self.models = {
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
        }
        self.feature_importance = None
        
    def create_features(self, df):
        """Create time-based and statistical features."""
        df = df.copy()
        
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(df)-1),
                periods=len(df),
                freq='h'
            )
            df.set_index('timestamp', inplace=True)
        
        # Extract numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Time-based features
        df_features = pd.DataFrame(index=df.index)
        df_features['hour'] = df.index.hour
        df_features['day_of_week'] = df.index.dayofweek
        df_features['month'] = df.index.month
        df_features['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Copy numeric columns
        for col in numeric_cols:
            df_features[col] = df[col]
        
        # Rolling statistics
        windows = [6, 12, 24]
        for window in windows:
            for col in numeric_cols:
                # Rolling mean
                df_features[f'{col}_rolling_mean_{window}h'] = (
                    df_features[col].rolling(window=window, min_periods=1)
                    .mean()
                    .fillna(method='bfill')
                    .fillna(method='ffill')
                )
                
                # Rolling std
                df_features[f'{col}_rolling_std_{window}h'] = (
                    df_features[col].rolling(window=window, min_periods=1)
                    .std()
                    .fillna(method='bfill')
                    .fillna(method='ffill')
                )
        
        # Lag features
        lags = [1, 3, 6, 12, 24]
        for lag in lags:
            for col in numeric_cols:
                df_features[f'{col}_lag_{lag}h'] = (
                    df_features[col].shift(lag)
                    .fillna(method='bfill')
                )
        
        return df_features.fillna(0)
    
    def prepare_data(self, df, target_col='C6H6(GT)'):
        """Prepare data for modeling with future target."""
        # Create features
        df_processed = self.create_features(df)
        
        # Ensure target column exists
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Shift target for future prediction
        y = df_processed[target_col].shift(-self.forecast_horizon)
        
        # Remove target from features
        X = df_processed.drop(columns=[target_col])
        
        # Remove rows with NaN targets
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def scale_features(self, X, training=True):
        """Scale features using StandardScaler."""
        if training:
            self.scalers = {}
            X_scaled = pd.DataFrame(index=X.index)
            
            for column in X.columns:
                self.scalers[column] = StandardScaler()
                X_scaled[column] = self.scalers[column].fit_transform(X[[column]]).ravel()
        else:
            X_scaled = pd.DataFrame(index=X.index)
            for column in X.columns:
                if column in self.scalers:
                    X_scaled[column] = self.scalers[column].transform(X[[column]]).ravel()
                else:
                    X_scaled[column] = X[column]
                    
        return X_scaled
    
    def train(self, X, y):
        """Train ensemble models."""
        X_scaled = self.scale_features(X, training=True)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            
        # Store feature importance from GBM
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['gbm'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        """Make ensemble predictions."""
        X_scaled = self.scale_features(X, training=False)
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
            
        # Ensemble prediction (weighted average)
        final_prediction = (0.7 * predictions['gbm'] + 0.3 * predictions['svr'])
        return final_prediction, predictions
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        final_pred, model_preds = self.predict(X)
        
        results = {
            'Ensemble MAE': mean_absolute_error(y, final_pred),
            'Ensemble R2': r2_score(y, final_pred)
        }
        
        for name, preds in model_preds.items():
            results[f'{name} MAE'] = mean_absolute_error(y, preds)
            results[f'{name} R2'] = r2_score(y, preds)
            
        return results
    
    def plot_feature_importance(self, top_n=10):
        """Plot top feature importance."""
        if self.feature_importance is None:
            print("No feature importance available. Train the model first.")
            return
            
        plt.figure(figsize=(12, 6))
        data = self.feature_importance.head(top_n)
        sns.barplot(data=data, x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def plot_predictions(self, y_true, y_pred, title='Actual vs Predicted Values'):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curve(self, X_train, y_train):
        """Plot learning curve of the Gradient Boosting model."""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.models['gbm'],
            X_train,
            y_train,
            cv=5,
            scoring='neg_mean_absolute_error',
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label="Training error")
        plt.plot(train_sizes, test_scores_mean, label="Cross-validation error")
        plt.title("Learning Curve: Gradient Boosting")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """Plot residuals (errors) for predictions."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(0, y_pred.min(), y_pred.max(), linestyles='dashed', colors='r')
        plt.title("Residual Plot: Predicted vs Residuals")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.grid(True)
        plt.show()
        
    def plot_time_series(self, y_true, y_pred, index):
        """Plot actual vs predicted values over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(index, y_true, label="Actual", color='blue', alpha=0.6)
        plt.plot(index, y_pred, label="Predicted", color='orange', alpha=0.6)
        plt.title("Actual vs Predicted Values Over Time")
        plt.xlabel("Time")
        plt.ylabel("Air Quality Index")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_error_distribution(self, y_true, y_pred):
        """Plot distribution of prediction errors (residuals)."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='purple')
        plt.title("Distribution of Prediction Errors (Residuals)")
        plt.xlabel("Residuals (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    file_path = 'AirQualityUCI.xlsx'  # Ensure the file path is correct
    data = pd.read_excel(file_path)
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = AirQualityPredictor(forecast_horizon=5)
    
    # Prepare data
    print("Preparing data...")
    X, y = predictor.prepare_data(data)
    
    # Initialize TimeSeriesSplit (this is the missing step)
    print("Performing cross-validation...")
    tscv = TimeSeriesSplit(n_splits=10)  # Ensure this initialization happens before the loop
    
    # Cross-validation loop
    cv_results = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {i}/10")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        print("Training models...")
        predictor.train(X_train, y_train)
        
        # Evaluate
        print("Evaluating performance...")
        results = predictor.evaluate(X_test, y_test)
        cv_results.append(results)
        
        # Make predictions
        final_pred, _ = predictor.predict(X_test)
        
        # Plot predictions for this fold
        predictor.plot_predictions(
            y_test,
            final_pred,
            title=f'Fold {i}: Actual vs Predicted Values (MAE: {results["Ensemble MAE"]:.3f})'
        )
        
        # Plot residuals for this fold
        predictor.plot_residuals(y_test, final_pred)
        
        # Plot time series of actual vs predicted
        predictor.plot_time_series(y_test, final_pred, index=X_test.index)
        
        # Plot error distribution for this fold
        predictor.plot_error_distribution(y_test, final_pred)
    
    # Learning curve (optional: you can plot it once for the whole dataset)
    predictor.plot_learning_curve(X_train, y_train)
    
    # Print average results
    print("\nAverage Cross-validation Results:")
    avg_results = pd.DataFrame(cv_results).mean()
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot feature importance
    predictor.plot_feature_importance()