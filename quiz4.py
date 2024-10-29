import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AirQualityPredictor:
    def __init__(self, forecast_horizon=5):
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        self.feature_importance = None
        self.models = {}
        
    def standardize(self, x):
        """Simple standardization (z-score normalization)"""
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std if std != 0 else 1), mean, std
    
    def create_features(self, df):
        """Create time-based and statistical features."""
        df = df.copy()
        
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(hours=len(df)-1),
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
        """Scale features using standardization."""
        if training:
            self.scalers = {}
            X_scaled = pd.DataFrame(index=X.index)
            
            for column in X.columns:
                scaled_values, mean, std = self.standardize(X[column].values)
                self.scalers[column] = {'mean': mean, 'std': std}
                X_scaled[column] = scaled_values
        else:
            X_scaled = pd.DataFrame(index=X.index)
            for column in X.columns:
                if column in self.scalers:
                    mean = self.scalers[column]['mean']
                    std = self.scalers[column]['std']
                    X_scaled[column] = (X[column] - mean) / (std if std != 0 else 1)
                else:
                    X_scaled[column] = X[column]
                    
        return X_scaled
    
    class SimpleGBM:
        """Simple Gradient Boosting implementation"""
        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.trees = []
            self.feature_importances_ = None
            
        def _calculate_residuals(self, y_true, y_pred):
            return y_true - y_pred
            
        def fit(self, X, y):
            self.feature_importances_ = np.zeros(X.shape[1])
            y_pred = np.zeros_like(y)
            
            for _ in range(self.n_estimators):
                residuals = self._calculate_residuals(y, y_pred)
                tree = self._build_simple_tree(X, residuals)
                self.trees.append(tree)
                
                # Update predictions
                predictions = self._predict_tree(X, tree)
                y_pred += self.learning_rate * predictions
                
                # Update feature importance
                self.feature_importances_ += np.abs(tree['split_feature_importance'])
                
            # Normalize feature importance
            self.feature_importances_ /= np.sum(self.feature_importances_)
            
        def _build_simple_tree(self, X, y):
            """Build a very simple decision tree"""
            best_feature = np.random.randint(0, X.shape[1])
            split_value = np.median(X[:, best_feature])
            
            left_mask = X[:, best_feature] <= split_value
            right_mask = ~left_mask
            
            tree = {
                'feature': best_feature,
                'split_value': split_value,
                'left_pred': np.mean(y[left_mask]) if np.any(left_mask) else 0,
                'right_pred': np.mean(y[right_mask]) if np.any(right_mask) else 0,
                'split_feature_importance': np.zeros(X.shape[1])
            }
            
            tree['split_feature_importance'][best_feature] = np.abs(tree['left_pred'] - tree['right_pred'])
            
            return tree
            
        def predict(self, X):
            y_pred = np.zeros(X.shape[0])
            
            for tree in self.trees:
                predictions = self._predict_tree(X, tree)
                y_pred += self.learning_rate * predictions
                
            return y_pred
            
        def _predict_tree(self, X, tree):
            predictions = np.zeros(X.shape[0])
            mask = X[:, tree['feature']] <= tree['split_value']
            predictions[mask] = tree['left_pred']
            predictions[~mask] = tree['right_pred']
            return predictions
    
    def train(self, X, y):
        """Train model."""
        X_scaled = self.scale_features(X, training=True)
        X_array = X_scaled.values
        
        self.models['gbm'] = self.SimpleGBM()
        print("Training GBM...")
        self.models['gbm'].fit(X_array, y.values)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models['gbm'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scale_features(X, training=False)
        X_array = X_scaled.values
        
        predictions = self.models['gbm'].predict(X_array)
        return predictions
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        # Calculate metrics
        mae = np.mean(np.abs(y - predictions))
        mse = np.mean((y - predictions) ** 2)
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
    
    def plot_feature_importance(self, top_n=10):
        """Plot top feature importance."""
        if self.feature_importance is None:
            print("No feature importance available. Train the model first.")
            return
            
        plt.figure(figsize=(12, 6))
        data = self.feature_importance.head(top_n)
        plt.bar(data['feature'], data['importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
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
        
    def plot_residuals(self, y_true, y_pred):
        """Plot residuals (errors) for predictions."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
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
        plt.hist(residuals, bins=30, density=True, alpha=0.7)
        plt.title("Distribution of Prediction Errors (Residuals)")
        plt.xlabel("Residuals (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    file_path = 'AirQualityUCI.xlsx'
    data = pd.read_excel(file_path)
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = AirQualityPredictor(forecast_horizon=5)
    
    # Prepare data
    print("Preparing data...")
    X, y = predictor.prepare_data(data)
    
    # Simple time series split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    print("Training models...")
    predictor.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Evaluate
    print("Evaluating performance...")
    results = predictor.evaluate(X_test, y_test)
    print("\nResults:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot various visualizations
    predictor.plot_predictions(y_test, predictions)
    predictor.plot_residuals(y_test, predictions)
    predictor.plot_time_series(y_test, predictions, X_test.index)
    predictor.plot_error_distribution(y_test, predictions)
    predictor.plot_feature_importance()
