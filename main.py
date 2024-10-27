import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Type
import random
import warnings
import time

from torch_activations import SigmoidActivation, TanhActivation, ReLUActivation, MishActivation, NCUActivation


class SyntheticStockDataGenerator:
    """Generates synthetic stock price data"""

    def __init__(self, n_stocks: int = 5, n_days: int = 1260, initial_price: float = 100.0):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.initial_price = initial_price

    def generate_prices(self) -> pd.DataFrame:
        """Generate synthetic daily stock prices"""
        # Parameters for the random walk
        drift = 0.0001
        volatility = 0.02

        # Generate random walks
        daily_returns = np.random.normal(drift, volatility, (self.n_days, self.n_stocks))
        price_paths = self.initial_price * np.exp(np.cumsum(daily_returns, axis=0))

        # Convert to DataFrame
        dates = pd.date_range(end='2024-01-01', periods=self.n_days, freq='B')
        df = pd.DataFrame(price_paths, index=dates,
                          columns=[f'Stock_{i+1}' for i in range(self.n_stocks)])
        return df


class TimeSeriesDataset:
    """Handles loading and preprocessing of various time series datasets"""

    @staticmethod
    def load_synthetic_stock_data():
        """Generate synthetic stock price data using SyntheticStockDataGenerator"""
        generator = SyntheticStockDataGenerator(n_stocks=5, n_days=1260, initial_price=100.0)
        df_prices = generator.generate_prices()
        return df_prices.mean(axis=1).values

    @staticmethod
    def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
        """Generate positional encodings"""
        position = np.arange(0, seq_length).reshape(seq_length, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        return pe

    @staticmethod
    def prepare_sequences(data: np.ndarray, seq_length: int = 10, d_model: int = 4) -> Tuple[np.ndarray, MinMaxScaler]:
        """Prepare data for rolling window prediction"""
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data.reshape(-1, 1))
        pe = TimeSeriesDataset.positional_encoding(seq_length, d_model)

        return data_normalized, pe, scaler

class TimeSeriesModel(nn.Module):
    """LSTM-based time series prediction model"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 activation_class: Type[nn.Module]):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.activation = activation_class()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        activated = self.activation(lstm_out[:, -1, :])
        return self.linear(activated)

class ModelTrainer:
    """Handles model training and evaluation with rolling window"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

    def create_batch_sequences(self, data: np.ndarray, indices: List[int], seq_length: int, pe: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a batch of sequences with positional encoding"""
        batch_size = len(indices)

        # Initialize tensors for batch
        X = np.zeros((batch_size, seq_length, 1 + pe.shape[1]))
        y = np.zeros((batch_size, 1))

        # Fill batch with sequences
        for i, idx in enumerate(indices):
            seq = data[idx:idx+seq_length]
            seq_with_pe = np.hstack((seq, pe))
            X[i] = seq_with_pe
            y[i] = data[idx + seq_length]

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def train_model(self,
                    model: nn.Module,
                    data: np.ndarray,
                    pe: np.ndarray,
                    seq_length: int,
                    batch_size: int = 32,
                    epochs: int = 100,
                    learning_rate: float = 0.001) -> Dict[str, List[float]]:

        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = {'train_loss': [], 'val_loss': []}
        n_train = int(0.7 * (len(data) - seq_length))
        n_val_start = n_train
        n_val_end = int(0.85 * (len(data) - seq_length))

        # Calculate number of batches per epoch
        train_indices = list(range(n_train))
        batches_per_epoch = (n_train + batch_size - 1) // batch_size

        for epoch in range(epochs):
            model.train()
            train_losses = []

            # Random shuffle for each epoch
            random.shuffle(train_indices)

            # Process batches
            for batch in range(batches_per_epoch):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = train_indices[start_idx:end_idx]

                # Create batch
                X, y = self.create_batch_sequences(data, batch_indices, seq_length, pe)
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                # Process validation set in batches
                val_indices = list(range(n_val_start, n_val_end))
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i:i + batch_size]
                    X, y = self.create_batch_sequences(data, batch_indices, seq_length, pe)
                    X, y = X.to(self.device), y.to(self.device)

                    y_pred = model(X)
                    val_loss = criterion(y_pred, y)
                    val_losses.append(val_loss.item())

            # Record metrics
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {history["train_loss"][-1]:.4f}, '
                      f'Val Loss = {history["val_loss"][-1]:.4f}')

        return history

    def evaluate_model(self,
                       model: nn.Module,
                       data: np.ndarray,
                       pe: np.ndarray,
                       seq_length: int,
                       scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model using rolling window prediction"""
        model.eval()
        test_start = int(0.85 * (len(data) - seq_length))
        predictions = []
        actuals = []

        with torch.no_grad():
            for i in range(test_start, len(data) - seq_length):
                # Create single sequence for testing (maintaining rolling window approach)
                X = np.zeros((1, seq_length, 1 + pe.shape[1]))
                seq = data[i:i+seq_length]
                seq_with_pe = np.hstack((seq, pe))
                X[0] = seq_with_pe
                X = torch.FloatTensor(X).to(self.device)

                y_actual = data[i + seq_length, 0]
                y_pred = model(X).cpu().numpy().reshape(-1)[0]

                predictions.append(y_pred)
                actuals.append(y_actual)

        # Transform predictions and actuals back to original scale
        predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).squeeze()
        actuals_inv = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).squeeze()

        return predictions_inv, actuals_inv

class ActivationComparison:
    """Manages the comparison of different activation functions"""

    def __init__(self):
        self.activation_functions = {
            'Sigmoid': SigmoidActivation,
            'Tanh': TanhActivation,
            'ReLU': ReLUActivation,
            'Mish': MishActivation,
            'NCU': NCUActivation
        }

        self.datasets = {
            'SyntheticStock': TimeSeriesDataset.load_synthetic_stock_data,
        }

        self.trainer = ModelTrainer()
        self.results = {}

    def run_comparison(self,
                       sequence_length: int = 10,
                       hidden_size: int = 64,
                       num_layers: int = 2,
                       batch_size: int = 32,
                       epochs: int = 100,
                       d_model: int = 4):
        """Run comparison of all activation functions on all datasets"""

        for dataset_name, dataset_loader in self.datasets.items():
            print(f"\nProcessing dataset: {dataset_name}")

            try:
                # Load and prepare data
                data = dataset_loader()
                data_normalized, pe, scaler = TimeSeriesDataset.prepare_sequences(
                    data, sequence_length, d_model=d_model
                )

                dataset_results = {}

                for act_name, act_class in self.activation_functions.items():
                    print(f"\nTraining with {act_name} activation")

                    # Initialize model
                    model = TimeSeriesModel(
                        input_size=1 + d_model,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        activation_class=act_class
                    )

                    try:
                        # Train model with rolling window
                        history = self.trainer.train_model(
                            model=model,
                            data=data_normalized,
                            pe=pe,
                            seq_length=sequence_length,
                            batch_size=batch_size,
                            epochs=epochs
                        )

                        # Evaluate model with rolling window
                        predictions, actuals = self.trainer.evaluate_model(
                            model=model,
                            data=data_normalized,
                            pe=pe,
                            seq_length=sequence_length,
                            scaler=scaler
                        )

                        # Calculate metrics
                        metrics = {
                            'mse': mean_squared_error(actuals, predictions),
                            'mae': mean_absolute_error(actuals, predictions),
                            'r2': r2_score(actuals, predictions),
                            'training_history': history,
                            'predictions': predictions,
                            'actuals': actuals
                        }

                        dataset_results[act_name] = metrics

                    except Exception as e:
                        print(f"Error training {act_name} on {dataset_name}: {e}")
                        continue

                self.results[dataset_name] = dataset_results

            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                continue

    def plot_results(self):
        """Create visualizations comparing activation function performance"""
        if not self.results:
            print("No results to plot!")
            return

        # 1. Training Loss Curves
        for dataset_name in self.results.keys():
            plt.figure(figsize=(15, 8))
            for act_name, metrics in self.results[dataset_name].items():
                history = metrics['training_history']
                plt.plot(history['train_loss'], label=f'{act_name} (Train)')
                plt.plot(history['val_loss'], label=f'{act_name} (Val)')

            plt.title(f'Training and Validation Loss - {dataset_name} Dataset')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        # 2. Comparison of Final Metrics
        metrics_to_plot = ['mse', 'mae', 'r2']

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, metric in enumerate(metrics_to_plot):
            data = []
            for dataset_name in self.results.keys():
                for act_name in self.results[dataset_name].keys():
                    data.append({
                        'Dataset': dataset_name,
                        'Activation': act_name,
                        'Value': self.results[dataset_name][act_name][metric]
                    })

            df = pd.DataFrame(data)
            sns.barplot(x='Activation', y='Value', data=df, ax=axes[i])
            axes[i].set_title(f'{metric.upper()}')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # 3. Predictions vs Actual Values (Detailed View)
        for dataset_name in self.results.keys():
            for act_name, metrics in self.results[dataset_name].items():
                # Create figure with secondary y-axis
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                fig.suptitle(f'{act_name} Activation on {dataset_name} Dataset', fontsize=16)

                # Get actual and predicted values
                actuals = metrics['actuals']
                predictions = metrics['predictions']

                # Calculate error
                error = np.abs(actuals - predictions)

                # Plot actual vs predicted
                ax1.plot(actuals, label='Actual', alpha=0.7)
                ax1.plot(predictions, label='Predicted', alpha=0.7)
                ax1.set_title('Actual vs Predicted Values')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Value')
                ax1.legend()

                # Plot error
                ax2.plot(error, label='Absolute Error', color='red', alpha=0.5)
                ax2.fill_between(range(len(error)), error, alpha=0.2, color='red')
                ax2.set_title('Prediction Error Over Time')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Absolute Error')
                ax2.legend()

                plt.tight_layout()
                plt.show()

                # 4. Scatter plot of predicted vs actual values
                plt.figure(figsize=(10, 10))
                plt.scatter(actuals, predictions, alpha=0.5)
                plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect Prediction')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'{act_name} - Predicted vs Actual Values')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                # 5. Zoom in on a subset of predictions
                window_size = min(100, len(actuals))  # Show last 100 points or all if less
                plt.figure(figsize=(15, 6))
                plt.plot(actuals[-window_size:], label='Actual', marker='o', markersize=4, alpha=0.7)
                plt.plot(predictions[-window_size:], label='Predicted', marker='o', markersize=4, alpha=0.7)
                plt.title(f'{act_name} - Detailed View of Last {window_size} Predictions')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

def main():
    """Run the complete activation function comparison"""

    comparison = ActivationComparison()

    # Run comparison with specified parameters
    comparison.run_comparison(
        sequence_length=20,
        hidden_size=10,
        num_layers=3,
        epochs=80,
        d_model=4
    )

    # Generate and display result plots
    comparison.plot_results()

if __name__ == "__main__":
    main()