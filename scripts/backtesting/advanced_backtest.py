import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    """
    Advanced backtesting system with realistic trading constraints.
    
    Features:
    - Transaction costs (commissions, slippage)
    - Market impact modeling
    - Position sizing and risk management
    - Realistic order execution
    - Portfolio-level constraints
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005,   # 0.05% slippage
                 max_position_size: float = 0.2,  # 20% max position
                 stop_loss_pct: float = 0.05,     # 5% stop loss
                 take_profit_pct: float = 0.10,   # 10% take profit
                 max_drawdown: float = 0.25,      # 25% max drawdown
                 rebalance_frequency: str = 'daily'):
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.rebalance_frequency = rebalance_frequency
        
        # Portfolio tracking
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.daily_returns = []
        self.max_portfolio_value = initial_capital
        
    def calculate_market_impact(self, order_size: float, avg_volume: float, price: float) -> float:
        """
        Calculate market impact of a trade.
        
        Args:
            order_size: Number of shares to trade
            avg_volume: Average daily volume
            price: Current price
        
        Returns:
            Market impact as a percentage
        """
        
        # Square root model for market impact
        volume_ratio = order_size / avg_volume
        market_impact = 0.1 * np.sqrt(volume_ratio)  # 10% impact at 100% of volume
        
        return market_impact
    
    def execute_trade(self, 
                     symbol: str,
                     action: str,  # 'buy' or 'sell'
                     shares: float,
                     price: float,
                     volume: float,
                     timestamp: pd.Timestamp) -> Dict:
        """
        Execute a trade with realistic constraints.
        
        Returns:
            Trade execution details
        """
        
        # Calculate market impact
        market_impact = self.calculate_market_impact(shares, volume, price)
        
        # Adjust price for market impact
        if action == 'buy':
            execution_price = price * (1 + market_impact + self.slippage_rate)
        else:  # sell
            execution_price = price * (1 - market_impact - self.slippage_rate)
        
        # Calculate transaction costs
        trade_value = execution_price * shares
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission
        
        # Check if we have enough cash for buy orders
        if action == 'buy' and total_cost > self.cash:
            # Adjust shares to fit available cash
            max_shares = (self.cash - commission) / execution_price
            shares = max_shares
            trade_value = execution_price * shares
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission
        
        # Update portfolio
        if action == 'buy':
            self.cash -= total_cost
            
            if symbol in self.positions:
                # Average down/up
                current_shares = self.positions[symbol]['shares']
                current_cost = self.positions[symbol]['cost']
                new_shares = current_shares + shares
                new_cost = current_cost + total_cost
                self.positions[symbol] = {
                    'shares': new_shares,
                    'cost': new_cost,
                    'avg_price': new_cost / new_shares,
                    'entry_time': timestamp
                }
            else:
                self.positions[symbol] = {
                    'shares': shares,
                    'cost': total_cost,
                    'avg_price': total_cost / shares,
                    'entry_time': timestamp
                }
        
        else:  # sell
            if symbol in self.positions:
                position = self.positions[symbol]
                shares_to_sell = min(shares, position['shares'])
                
                if shares_to_sell > 0:
                    self.cash += (execution_price * shares_to_sell) - commission
                    
                    # Update position
                    remaining_shares = position['shares'] - shares_to_sell
                    if remaining_shares > 0:
                        self.positions[symbol]['shares'] = remaining_shares
                        self.positions[symbol]['cost'] = position['cost'] * (remaining_shares / position['shares'])
                    else:
                        del self.positions[symbol]
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'market_impact': market_impact,
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value(price)
        }
        
        self.trade_history.append(trade_record)
        
        return trade_record
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        
        position_value = sum(
            pos['shares'] * current_price 
            for pos in self.positions.values()
        )
        
        return self.cash + position_value
    
    def check_risk_limits(self, current_price: float) -> List[Dict]:
        """
        Check and return positions that violate risk limits.
        
        Returns:
            List of positions to close
        """
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_value = position['shares'] * current_price
            position_pct = current_value / self.portfolio_value
            
            # Check position size limit
            if position_pct > self.max_position_size:
                positions_to_close.append({
                    'symbol': symbol,
                    'reason': 'position_size_limit',
                    'shares': position['shares']
                })
                continue
            
            # Check stop loss
            loss_pct = (current_price - position['avg_price']) / position['avg_price']
            if loss_pct < -self.stop_loss_pct:
                positions_to_close.append({
                    'symbol': symbol,
                    'reason': 'stop_loss',
                    'shares': position['shares']
                })
                continue
            
            # Check take profit
            if loss_pct > self.take_profit_pct:
                positions_to_close.append({
                    'symbol': symbol,
                    'reason': 'take_profit',
                    'shares': position['shares']
                })
        
        return positions_to_close
    
    def run_backtest(self, 
                    df: pd.DataFrame,
                    predictions: List[float],
                    confidence_threshold: float = 0.6) -> Dict:
        """
        Run advanced backtest with realistic constraints.
        
        Args:
            df: Price data with OHLCV
            predictions: Model predictions (-1 to 1)
            confidence_threshold: Minimum confidence to trade
        
        Returns:
            Backtest results
        """
        
        print("Running advanced backtest...")
        
        # Initialize tracking
        portfolio_values = [self.initial_capital]
        daily_returns = []
        trades_executed = 0
        
        for i in range(len(df)):
            if i < 20:  # Skip first 20 days for feature calculation
                continue
            
            current_data = df.iloc[i]
            current_price = current_data['Close']
            current_volume = current_data['Volume']
            timestamp = df.index[i]
            prediction = predictions[i] if i < len(predictions) else 0
            
            # Update portfolio value
            self.portfolio_value = self.get_portfolio_value(current_price)
            portfolio_values.append(self.portfolio_value)
            
            # Check for new maximum
            if self.portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = self.portfolio_value
            
            # Check drawdown limit
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.max_drawdown:
                print(f"Max drawdown exceeded at {timestamp}. Stopping backtest.")
                break
            
            # Check risk limits and close positions if needed
            positions_to_close = self.check_risk_limits(current_price)
            for close_info in positions_to_close:
                symbol = close_info['symbol']
                shares = close_info['shares']
                reason = close_info['reason']
                
                self.execute_trade(
                    symbol=symbol,
                    action='sell',
                    shares=shares,
                    price=current_price,
                    volume=current_volume,
                    timestamp=timestamp
                )
                trades_executed += 1
                print(f"Closed position in {symbol} due to {reason}")
            
            # Generate new signals
            if abs(prediction) > confidence_threshold:
                action = 'buy' if prediction > 0 else 'sell'
                
                # Calculate position size based on Kelly Criterion
                win_rate = 0.55  # Estimated win rate
                avg_win = 0.02   # Average win
                avg_loss = 0.01  # Average loss
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                
                # Adjust for prediction confidence
                position_size = kelly_fraction * abs(prediction) * self.portfolio_value / current_price
                
                # Apply position size limits
                max_shares = self.portfolio_value * self.max_position_size / current_price
                position_size = min(position_size, max_shares)
                
                if position_size > 0:
                    # Execute trade
                    trade_result = self.execute_trade(
                        symbol='STOCK',  # Assuming single stock
                        action=action,
                        shares=position_size,
                        price=current_price,
                        volume=current_volume,
                        timestamp=timestamp
                    )
                    trades_executed += 1
            
            # Calculate daily return
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)
        
        # Calculate final metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio
        if daily_returns:
            returns_series = pd.Series(daily_returns)
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate other metrics
        win_rate = len([t for t in self.trade_history if t.get('pnl', 0) > 0]) / len(self.trade_history) if self.trade_history else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': trades_executed,
            'portfolio_values': portfolio_values,
            'trade_history': self.trade_history,
            'daily_returns': daily_returns
        }
        
        return results

def calculate_advanced_metrics(results: Dict) -> Dict:
    """Calculate additional performance metrics."""
    
    portfolio_values = results['portfolio_values']
    daily_returns = results['daily_returns']
    
    # Convert to pandas series for easier calculations
    portfolio_series = pd.Series(portfolio_values)
    returns_series = pd.Series(daily_returns)
    
    # Basic metrics
    total_return = results['total_return']
    sharpe_ratio = results['sharpe_ratio']
    max_drawdown = results['max_drawdown']
    
    # Additional metrics
    volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
    var_95 = returns_series.quantile(0.05) if len(returns_series) > 0 else 0
    cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series) > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (return / downside deviation)
    downside_returns = returns_series[returns_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (returns_series.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
    
    # Information ratio (excess return / tracking error)
    # Assuming risk-free rate of 2%
    risk_free_rate = 0.02
    excess_return = total_return - risk_free_rate
    tracking_error = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
    
    # Maximum consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    for ret in daily_returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    # Recovery time (days to recover from max drawdown)
    recovery_time = 0
    if max_drawdown < 0:
        peak_idx = portfolio_series.idxmax()
        recovery_idx = portfolio_series[portfolio_series.index > peak_idx]
        recovery_idx = recovery_idx[recovery_idx >= portfolio_series.max()]
        if len(recovery_idx) > 0:
            recovery_time = (recovery_idx.index[0] - peak_idx).days
    
    advanced_metrics = {
        'volatility': volatility,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'information_ratio': information_ratio,
        'max_consecutive_losses': max_consecutive_losses,
        'recovery_time_days': recovery_time,
        'risk_adjusted_return': total_return / volatility if volatility > 0 else 0
    }
    
    return {**results, **advanced_metrics}

def main():
    """Example usage of advanced backtesting."""
    
    print("Advanced Backtesting System")
    print("This system includes:")
    print("- Realistic transaction costs (commissions, slippage)")
    print("- Market impact modeling")
    print("- Dynamic position sizing")
    print("- Risk management (stop-loss, take-profit)")
    print("- Portfolio-level constraints")
    print("- Advanced performance metrics")
    
    print("\nKey features:")
    print("- Kelly Criterion position sizing")
    print("- Market impact calculation")
    print("- Drawdown monitoring")
    print("- Risk-adjusted performance metrics")
    print("- Detailed trade tracking")

if __name__ == "__main__":
    main() 