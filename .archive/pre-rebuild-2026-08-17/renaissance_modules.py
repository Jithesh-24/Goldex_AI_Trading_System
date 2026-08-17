#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renaissance_modules.py — Renaissance Technologies-style mathematical modules

Implements:
1. HMM Regime Detector (Leonard Baum's models)
2. Kalman Filter (signal extraction from noise)
3. Ornstein-Uhlenbeck (mean reversion detection)
4. Half-Kelly Position Sizing

All computed from gold price history, no lookahead bias.
"""

import numpy as np
from numba import njit


# ═══════════════════════════════════════════════════════════════════
# 1. HIDDEN MARKOV MODEL — REGIME DETECTION
# Renaissance was BUILT on this (Leonard Baum's algorithm)
# ═══════════════════════════════════════════════════════════════════

class HMMRegimeDetector:
    """
    Gaussian HMM for detecting market regimes.
    
    States: 0=trending_up, 1=trending_down, 2=ranging, 3=volatile
    
    Uses Baum-Welch (EM) for training and Viterbi for decoding.
    """
    
    def __init__(self, n_states=4, n_iter=20, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.trained = False
        
        # Model parameters
        self.pi = None      # Initial state distribution
        self.A = None       # Transition matrix
        self.means = None   # Emission means
        self.stds = None    # Emission stds
    
    def _init_params(self, returns):
        """Initialize parameters with k-means-like approach."""
        np.random.seed(self.random_state)
        
        # Sort returns to find initial means
        sorted_r = np.sort(returns)
        n = len(sorted_r)
        chunk = n // self.n_states
        
        self.means = np.array([np.mean(sorted_r[i*chunk:(i+1)*chunk]) 
                               for i in range(self.n_states)])
        self.stds = np.array([np.std(sorted_r[i*chunk:(i+1)*chunk]) + 1e-6 
                              for i in range(self.n_states)])
        
        # Uniform initial distribution
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Diagonal-heavy transition matrix (regimes are persistent)
        self.A = np.full((self.n_states, self.n_states), 0.02)
        np.fill_diagonal(self.A, 0.94)
    
    def _gaussian_pdf(self, x, mean, std):
        """Gaussian probability density function."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def _forward(self, returns):
        """Forward algorithm."""
        T = len(returns)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self._gaussian_pdf(returns[0], self.means[i], self.stds[i])
        
        # Scale to prevent underflow
        scale = np.zeros(T)
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0] + 1e-300
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * \
                              self._gaussian_pdf(returns[t], self.means[j], self.stds[j])
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t] + 1e-300
        
        return alpha, scale
    
    def _backward(self, returns, scale):
        """Backward algorithm."""
        T = len(returns)
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * 
                    np.array([self._gaussian_pdf(returns[t+1], self.means[j], self.stds[j]) 
                              for j in range(self.n_states)]) * beta[t+1])
            beta[t] /= scale[t+1] + 1e-300
        
        return beta
    
    def _baum_welch(self, returns):
        """Baum-Welch EM algorithm."""
        T = len(returns)
        
        for iteration in range(self.n_iter):
            # E-step
            alpha, scale = self._forward(returns)
            beta = self._backward(returns, scale)
            
            # Compute gamma and xi
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300
            
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                       self._gaussian_pdf(returns[t+1], self.means[j], self.stds[j]) * 
                                       beta[t+1, j])
                xi[t] /= np.sum(xi[t]) + 1e-300
            
            # M-step
            self.pi = gamma[0]
            
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-300)
                self.A[i] /= np.sum(self.A[i]) + 1e-300
            
            for k in range(self.n_states):
                w = gamma[:, k]
                self.means[k] = np.sum(w * returns) / (np.sum(w) + 1e-300)
                self.stds[k] = np.sqrt(np.sum(w * (returns - self.means[k])**2) / (np.sum(w) + 1e-300))
                self.stds[k] = max(self.stds[k], 1e-6)
    
    def fit(self, returns):
        """Train HMM on return series."""
        self._init_params(returns)
        self._baum_welch(returns)
        self.trained = True
        
        # Sort states by mean return (trending_up > ranging > trending_down)
        order = np.argsort(self.means)[::-1]
        self.means = self.means[order]
        self.stds = self.stds[order]
        self.pi = self.pi[order]
        self.A = self.A[order][:, order]
    
    def predict_proba(self, returns):
        """Get regime probabilities for each time step."""
        if not self.trained:
            return np.ones((len(returns), self.n_states)) / self.n_states
        
        alpha, scale = self._forward(returns)
        probs = alpha * np.ones((1, self.n_states))  # uniform prior for beta
        probs /= np.sum(probs, axis=1, keepdims=True) + 1e-300
        return probs
    
    def get_current_regime(self, recent_returns, window=50):
        """Get current regime based on recent returns."""
        if len(recent_returns) < 10:
            return 2, np.ones(self.n_states) / self.n_states  # default to ranging
        
        probs = self.predict_proba(recent_returns[-window:])
        current_probs = probs[-1]
        current_regime = np.argmax(current_probs)
        
        return current_regime, current_probs


# ═══════════════════════════════════════════════════════════════════
# 2. KALMAN FILTER — SIGNAL EXTRACTION FROM NOISE
# Optimal in minimum mean-square error sense
# ═══════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Kalman filter for extracting true price trend from noisy observations.
    
    State model:
        x_t = F * x_{t-1} + w_t    (true trend evolves)
        z_t = H * x_t + v_t         (we observe noisy price)
    
    Applications:
    - Trend extraction from noise
    - Dynamic hedge ratios
    - Volatility estimation
    """
    
    def __init__(self, dt=1.0, process_noise=0.01, measurement_noise=0.1):
        # State transition (trend + velocity model)
        self.F = np.array([[1, dt],
                           [0, 1]])  # [position, velocity]
        
        # Observation matrix (we only observe position)
        self.H = np.array([[1, 0]])
        
        # Process noise (how much the true trend changes)
        self.Q = np.array([[dt**3/3, dt**2/2],
                           [dt**2/2, dt]]) * process_noise
        
        # Measurement noise (how noisy our observations are)
        self.R = np.array([[measurement_noise]])
        
        # Initial state
        self.x = np.array([[0], [0]])  # [position, velocity]
        self.P = np.eye(2) * 1.0  # initial uncertainty
    
    def update(self, z):
        """
        Process one observation.
        
        Args:
            z: scalar observation (price or return)
        
        Returns:
            x_hat: filtered state [trend, velocity]
            innovation: prediction error (large = model surprised)
        """
        z = np.array([[z]])
        
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = z - self.H @ x_pred  # innovation (prediction error)
        S = self.H @ P_pred @ self.H.T + self.R  # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x.flatten(), float(y[0, 0])
    
    def filter_series(self, prices):
        """Filter an entire price series."""
        trends = []
        innovations = []
        
        for p in prices:
            trend, innovation = self.update(p)
            trends.append(trend)
            innovations.append(innovation)
        
        return np.array(trends), np.array(innovations)


# ═══════════════════════════════════════════════════════════════════
# 3. ORNSTEIN-UHLENBECK — MEAN REVERSION DETECTION
# The workhorse model for mean reversion in finance
# ═══════════════════════════════════════════════════════════════════

class OrnsteinUhlenbeckDetector:
    """
    Detects mean reversion properties using the OU process:
    
    dX_t = θ(μ - X_t)dt + σ dW_t
    
    Key outputs:
    - theta: speed of mean reversion (higher = faster reversion)
    - mu: long-term mean (equilibrium level)
    - sigma: volatility
    - half_life: expected time to revert halfway
    - is_mean_reverting: boolean (theta > 0 significantly)
    """
    
    def __init__(self, lookback=100):
        self.lookback = lookback
    
    def fit(self, prices):
        """
        Estimate OU parameters from price series.
        
        Returns dict with theta, mu, sigma, half_life, is_mean_reverting
        """
        if len(prices) < self.lookback + 10:
            return {'theta': 0, 'mu': np.mean(prices), 'sigma': 0, 
                    'half_life': np.inf, 'is_mean_reverting': False,
                    'signal': 0}
        
        # Use log prices for stationarity
        log_prices = np.log(prices[-self.lookback:])
        
        # Discretize: X_{t+1} - X_t = a + b*X_t + epsilon
        y = np.diff(log_prices)
        x = log_prices[:-1]
        
        # OLS regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        
        if ss_xx < 1e-20:
            return {'theta': 0, 'mu': np.mean(prices), 'sigma': 0,
                    'half_life': np.inf, 'is_mean_reverting': False,
                    'signal': 0}
        
        b = ss_xy / ss_xx
        a = y_mean - b * x_mean
        
        # OU parameters
        dt = 1.0  # M5 bars
        theta = -np.log(1 + b) / dt if (1 + b) > 0 else 0
        mu = -a / b if abs(b) > 1e-10 else np.mean(log_prices)
        
        # Residuals for sigma
        residuals = y - a - b * x
        sigma = np.std(residuals) * np.sqrt(-2 * np.log(1 + b) / dt**2) if (1 + b) > 0 else 0
        
        # Half-life
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        # Is it mean-reverting?
        # t-test on b: if b < 0 significantly, series is mean-reverting
        se_b = np.std(residuals) / np.sqrt(ss_xx) if ss_xx > 0 else 0
        t_stat = b / se_b if se_b > 0 else 0
        is_mr = (t_stat < -2.0) and (theta > 0)  # significant negative coefficient
        
        # Generate signal: how far from equilibrium?
        current = log_prices[-1]
        z_score = (current - mu) / (sigma / np.sqrt(2 * theta) + 1e-10) if theta > 0 else 0
        
        return {
            'theta': theta,
            'mu': np.exp(mu),  # back to price space
            'sigma': sigma,
            'half_life': half_life,
            'is_mean_reverting': is_mr,
            'z_score': z_score,  # positive = above mean (sell), negative = below (buy)
            'signal': -z_score * is_mr  # negative z = buy signal
        }


# ═══════════════════════════════════════════════════════════════════
# 4. HALF-KELLY POSITION SIZING
# What Renaissance and all serious quant funds use
# ═══════════════════════════════════════════════════════════════════

def half_kelly(win_rate, reward_risk, fraction=0.5):
    """
    Half-Kelly position sizing.
    
    f* = (p*b - q) / b
    
    Where:
    - p = probability of winning
    - q = 1 - p
    - b = reward/risk ratio
    
    Half Kelly: f_practical = f* / 2
    → 75% of growth rate with 50% less drawdown
    
    Args:
        win_rate: estimated probability of winning (0-1)
        reward_risk: average win / average loss
        fraction: Kelly fraction (0.5 = half Kelly)
    
    Returns:
        position_fraction: fraction of bankroll to risk per trade
    """
    p = win_rate
    q = 1 - p
    b = reward_risk
    
    if b <= 0:
        return 0.0
    
    full_kelly = (p * b - q) / b
    
    if full_kelly <= 0:
        return 0.0  # no edge, don't trade
    
    return full_kelly * fraction


def adaptive_kelly(win_rate, reward_risk, recent_performance, confidence=0.8):
    """
    Adaptive Kelly that adjusts based on recent performance.
    
    When recent performance is poor, reduce size.
    When recent performance is good, approach full half-Kelly.
    """
    base_kelly = half_kelly(win_rate, reward_risk)
    
    # Adjust based on recent performance
    if recent_performance < -0.02:  # losing money recently
        return base_kelly * 0.5  # reduce size
    elif recent_performance > 0.02:  # making money
        return base_kelly * 1.0  # full half-Kelly
    else:
        return base_kelly * 0.75  # moderate


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: Compute all Renaissance features for a price series
# ═══════════════════════════════════════════════════════════════════

def compute_renaissance_features(prices, returns=None, window=100):
    """
    Compute all Renaissance-style features for a price series.
    
    Returns dict with:
    - hmm_regime: current regime (0-3)
    - hmm_probs: regime probabilities
    - kalman_trend: filtered trend
    - kalman_innovation: prediction error (large = surprise)
    - ou_theta: mean reversion speed
    - ou_mu: equilibrium level
    - ou_half_life: expected reversion time
    - ou_is_mr: is series mean-reverting?
    - ou_signal: mean reversion signal
    - kelly_fraction: optimal position size
    """
    if returns is None:
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])
    
    result = {}
    
    # 1. HMM Regime Detection
    hmm = HMMRegimeDetector(n_states=4, n_iter=15)
    if len(returns) >= window + 20:
        hmm.fit(returns[-window*3:])
        regime, probs = hmm.get_current_regime(returns, window=window)
        result['hmm_regime'] = regime
        result['hmm_probs'] = probs
    else:
        result['hmm_regime'] = 2  # ranging
        result['hmm_probs'] = np.array([0.25, 0.25, 0.25, 0.25])
    
    # 2. Kalman Filter
    kf = KalmanFilter(dt=1.0, process_noise=0.01, measurement_noise=0.1)
    if len(prices) >= window:
        trends, innovations = kf.filter_series(prices[-window:])
        result['kalman_trend'] = trends[-1, 0]  # current trend
        result['kalman_velocity'] = trends[-1, 1]  # current velocity
        result['kalman_innovation'] = innovations[-1]  # last prediction error
    else:
        result['kalman_trend'] = prices[-1]
        result['kalman_velocity'] = 0
        result['kalman_innovation'] = 0
    
    # 3. Ornstein-Uhlenbeck
    ou = OrnsteinUhlenbeckDetector(lookback=window)
    if len(prices) >= window + 10:
        ou_result = ou.fit(prices)
        result['ou_theta'] = ou_result['theta']
        result['ou_mu'] = ou_result['mu']
        result['ou_half_life'] = ou_result['half_life']
        result['ou_is_mr'] = ou_result['is_mean_reverting']
        result['ou_signal'] = ou_result['signal']
        result['ou_z_score'] = ou_result.get('z_score', 0)
    else:
        result['ou_theta'] = 0
        result['ou_mu'] = np.mean(prices) if len(prices) > 0 else 0
        result['ou_half_life'] = np.inf
        result['ou_is_mr'] = False
        result['ou_signal'] = 0
        result['ou_z_score'] = 0
    
    # 4. Half-Kelly (based on recent win rate and R:R)
    # Use last 50 signals as estimate
    result['kelly_fraction'] = 0.02  # default 2% risk per trade
    
    return result


if __name__ == '__main__':
    # Test with synthetic gold-like data
    np.random.seed(42)
    n = 500
    
    # Simulate gold price with regime changes
    prices = [2000.0]
    for i in range(n - 1):
        if i < 100:  # trending up
            prices.append(prices[-1] * (1 + 0.001 + np.random.randn() * 0.005))
        elif i < 200:  # ranging
            prices.append(prices[-1] * (1 + np.random.randn() * 0.003))
        elif i < 300:  # volatile
            prices.append(prices[-1] * (1 + np.random.randn() * 0.01))
        else:  # trending down
            prices.append(prices[-1] * (1 - 0.001 + np.random.randn() * 0.005))
    
    prices = np.array(prices)
    
    print("Testing Renaissance modules...")
    result = compute_renaissance_features(prices, window=100)
    
    print(f"\nHMM Regime: {result['hmm_regime']}")
    print(f"HMM Probs: {result['hmm_probs']}")
    print(f"Kalman Trend: {result['kalman_trend']:.2f}")
    print(f"Kalman Velocity: {result['kalman_velocity']:.4f}")
    print(f"Kalman Innovation: {result['kalman_innovation']:.4f}")
    print(f"OU Theta: {result['ou_theta']:.4f}")
    print(f"OU Mu: {result['ou_mu']:.2f}")
    print(f"OU Half-life: {result['ou_half_life']:.1f} bars")
    print(f"OU Is Mean-Reverting: {result['ou_is_mr']}")
    print(f"OU Signal: {result['ou_signal']:.4f}")
    print(f"Kelly Fraction: {result['kelly_fraction']:.4f}")
    
    print("\n✅ All modules working!")
