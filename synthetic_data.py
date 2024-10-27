# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import johnsonsu
from statsmodels.distributions.copula.api import GaussianCopula
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class SyntheticStockDataGenerator:
    """
    Generates synthetic stock price data with complex characteristics including
    non-stationarity, heteroscedasticity, jumps, regime shifts, and cyclical components.
    """

    def __init__(self, n_stocks=5, n_days=1260, initial_price=100.0):
        """
        Initialize the generator with basic parameters.

        Args:
            n_stocks (int): Number of stocks to simulate.
            n_days (int): Number of trading days (approx. 5 years).
            initial_price (float): Starting price for all stocks.
        """
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.initial_price = initial_price

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Drift parameters (mean returns)
        self.drift = np.random.normal(0.0001, 0.0005, self.n_stocks)

        # GARCH(1,1) parameters
        self.omega = np.random.uniform(1e-6, 1e-5, self.n_stocks)
        self.alpha = np.random.uniform(0.05, 0.15, self.n_stocks)
        self.beta = np.random.uniform(0.8, 0.9, self.n_stocks)

        # Johnson SU parameters
        self.gamma = np.random.uniform(-1, 1, self.n_stocks)   # Shape parameter gamma, reduced for smoother tails
        self.delta = np.random.uniform(0.8, 1.5, self.n_stocks)  # Shape parameter delta, adjusted for tail behavior

        # Jump diffusion parameters
        self.jump_intensity = np.random.uniform(0.01, 0.03, self.n_stocks)  # Lower jump intensity for reduced tail spikes
        self.jump_scale = np.random.uniform(0.005, 0.02, self.n_stocks)      # Scale for jump size distribution

        # Regime-switching parameters
        self.num_regimes = 2
        self.regime_transition_matrix = np.array([[0.95, 0.05],
                                                  [0.05, 0.95]])
        self.regime_drifts = [0.0001, 0.0005]
        self.regime_volatilities = [0.01, 0.03]

        # Cyclical components (stock-specific amplitudes)
        self.daily_amplitude = np.random.uniform(0.0001, 0.0005, self.n_stocks)
        self.monthly_amplitude = np.random.uniform(0.001, 0.002, self.n_stocks)
        self.quarterly_amplitude = np.random.uniform(0.003, 0.005, self.n_stocks)
        self.yearly_amplitude = np.random.uniform(0.008, 0.01, self.n_stocks)

        # Covariance matrix and correlation matrix
        self.covariance = self._generate_covariance_matrix()
        self.correlation = self._cov_to_corr(self.covariance)

    def _generate_covariance_matrix(self):
        """Generate a positive definite covariance matrix."""
        A = np.random.rand(self.n_stocks, self.n_stocks)
        cov_matrix = np.dot(A, A.T)
        return cov_matrix

    def _cov_to_corr(self, cov_matrix):
        """Convert covariance matrix to correlation matrix."""
        D = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(D, D)
        corr_matrix[np.diag_indices_from(corr_matrix)] = 1.0
        return corr_matrix

    def _simulate_regime_switches(self):
        """Simulate regime switches using a Markov chain."""
        regimes = np.zeros(self.n_days, dtype=int)
        regimes[0] = np.random.choice(self.num_regimes)
        for t in range(1, self.n_days):
            regimes[t] = np.random.choice(
                self.num_regimes,
                p=self.regime_transition_matrix[regimes[t-1]]
            )
        self.regimes = regimes  # Store the regimes
        return regimes

    def _calculate_cyclical_component(self, t, s):
        """
        Calculate cyclical components for a given time point and stock.

        Args:
            t (int): Time index.
            s (int): Stock index.

        Returns:
            float: Combined cyclical effect.
        """
        daily = self.daily_amplitude[s] * np.sin(2 * np.pi * t / 1)
        monthly = self.monthly_amplitude[s] * np.sin(2 * np.pi * t / 21)
        quarterly = self.quarterly_amplitude[s] * np.sin(2 * np.pi * t / 63)
        yearly = self.yearly_amplitude[s] * np.sin(2 * np.pi * t / 252)
        return daily + monthly + quarterly + yearly

    def _simulate_jumps(self):
        """Simulate jumps using a compound Poisson process with heavy-tailed jump sizes."""
        jumps = np.zeros((self.n_days, self.n_stocks))
        for s in range(self.n_stocks):
            # Simulate number of jumps per day
            num_jumps = np.random.poisson(self.jump_intensity[s], self.n_days)
            for t in range(self.n_days):
                if num_jumps[t] > 0:
                    # Sum of jump sizes (heavy-tailed distribution)
                    jump_sizes = np.random.laplace(0, self.jump_scale[s], num_jumps[t])
                    jumps[t, s] = np.sum(jump_sizes)
        return jumps

    def _simulate_garch_volatility(self):
        """Simulate GARCH(1,1) volatility process."""
        volatilities = np.zeros((self.n_days, self.n_stocks))
        volatilities[0] = np.sqrt(self.omega / (1 - self.alpha - self.beta))
        for t in range(1, self.n_days):
            residuals = self.residuals[t-1]
            volatilities[t] = np.sqrt(
                self.omega +
                self.alpha * residuals**2 +
                self.beta * volatilities[t-1]**2
            )
        return volatilities

    def _generate_correlated_random_variables(self):
        """Generate correlated random variables using a Gaussian copula."""
        copula = GaussianCopula(corr=self.correlation)
        u = copula.rvs(nobs=self.n_days)
        return u

    def _simulate_returns(self):
        """Simulate returns incorporating all components."""
        # Simulate regimes if not already simulated
        if not hasattr(self, 'regimes'):
            self._simulate_regime_switches()
        regimes = self.regimes

        volatilities = np.zeros((self.n_days, self.n_stocks))
        returns = np.zeros((self.n_days, self.n_stocks))
        self.residuals = np.zeros((self.n_days, self.n_stocks))
        jumps = self._simulate_jumps()

        # Initial volatilities
        volatilities[0] = np.sqrt(self.omega / (1 - self.alpha - self.beta))

        # Generate correlated random variables
        u = self._generate_correlated_random_variables()
        # Ensure u is in (0,1)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        for t in range(1, self.n_days):
            for s in range(self.n_stocks):
                # GARCH volatility update
                residual = self.residuals[t-1, s]
                volatilities[t, s] = np.sqrt(
                    self.omega[s] +
                    self.alpha[s] * residual**2 +
                    self.beta[s] * volatilities[t-1, s]**2
                )

                # Regime-specific drift and volatility
                regime = regimes[t]
                drift = self.regime_drifts[regime]
                regime_volatility = self.regime_volatilities[regime]

                # Cyclical component
                cyclical = self._calculate_cyclical_component(t, s)

                # Inverse transform sampling for Johnson SU distribution
                gamma = self.gamma[s]
                delta = self.delta[s]
                # Convert uniform variable to Johnson SU distributed variable
                eps = johnsonsu.ppf(u[t, s], a=gamma, b=delta, loc=0, scale=1)
                scaled_eps = eps * volatilities[t, s] * regime_volatility

                # Total return
                total_return = drift + scaled_eps + jumps[t, s] + cyclical
                returns[t, s] = total_return
                self.residuals[t, s] = scaled_eps

        return returns

    def generate_prices(self):
        """Generate synthetic stock prices."""
        returns = self._simulate_returns()
        # Prices are exponentiated cumulative returns
        prices = self.initial_price * np.exp(np.cumsum(returns, axis=0))
        # Create DataFrame
        dates = pd.date_range(start=datetime(2015, 1, 1), periods=self.n_days, freq='B')
        df = pd.DataFrame(prices, index=dates, columns=[f'Stock_{i+1}' for i in range(self.n_stocks)])
        return df

    def generate_target_variable(self):
        """Generate a challenging target variable based on regime shifts."""
        # Ensure regimes are simulated
        if not hasattr(self, 'regimes'):
            self._simulate_regime_switches()
        regimes = self.regimes
        target = regimes[1:]  # Next day's regime
        # Align target length with data
        target = np.append(target, target[-1])
        return target

    def validate_data(self, returns):
        """Validate statistical properties of the simulated data."""
        for s in range(self.n_stocks):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            pd.plotting.autocorrelation_plot(returns[:, s])
            plt.title(f'Autocorrelation - Stock {s+1}')
            plt.subplot(1, 2, 2)
            pd.Series(returns[:, s]).hist(bins=50)
            plt.title(f'Returns Distribution - Stock {s+1}')
            plt.show()

def main():
    """Main function to generate data and save to CSV."""
    generator = SyntheticStockDataGenerator(n_stocks=5, n_days=1260, initial_price=100.0)
    df_prices = generator.generate_prices()
    target = generator.generate_target_variable()

    # Combine prices and target into a single DataFrame
    df = df_prices.copy()
    df['Target'] = target

    # Display summary statistics
    print("\nGenerated Data Summary:")
    print(df.describe())

    # Save to CSV
    df.to_csv('synthetic_stock_data.csv')
    print("\nData saved to 'synthetic_stock_data.csv'")

    # Optional: Validate statistical properties
    # Calculate returns
    returns = np.log(df_prices.values[1:] / df_prices.values[:-1])
    generator.validate_data(returns)

if __name__ == "__main__":
    main()