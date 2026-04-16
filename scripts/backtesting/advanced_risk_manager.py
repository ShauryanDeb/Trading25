import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """Advanced risk management system for trading models."""
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk per trade
                 max_position_size: float = 0.1,    # 10% max position size
                 stop_loss_pct: float = 0.05,       # 5% stop loss
                 trailing_stop: bool = True,
                 volatility_lookback: int = 20):
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop = trailing_stop
        self.volatility_lookback = volatility_lookback
        
        # Track positions and performance
        self.positions = {}
        self.portfolio_value = 100000  # Starting portfolio value
        self.trade_history = []
        
    def calculate_position_size(self, 
                              signal_strength: float, 
                              volatility: float, 
                              price: float,
                              confidence: float = 1.0) -> float:
        """
        Calculate optimal position size based on Kelly Criterion and risk management.
        
        Args:
            signal_strength: Model prediction confidence (-1 to 1)
            volatility: Current volatility estimate
            price: Current asset price
            confidence: Model confidence score (0 to 1)
        """
        
        # Kelly Criterion for position sizing
        win_rate = 0.55  # Estimated win rate
        avg_win = 0.02   # Average win percentage
        avg_loss = 0.01  # Average loss percentage
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Adjust for signal strength and confidence
        adjusted_kelly = kelly_fraction * abs(signal_strength) * confidence
        
        # Volatility adjustment (reduce position size in high volatility)
        volatility_adj = 1 / (1 + volatility * 10)  # Reduce size when volatility is high
        
        # Risk-based position sizing
        risk_per_share = price * self.stop_loss_pct
        max_risk_amount = self.portfolio_value * self.max_portfolio_risk
        max_shares_by_risk = max_risk_amount / risk_per_share
        
        # Final position size
        position_size = min(
            adjusted_kelly * self.portfolio_value / price,
            max_shares_by_risk,
            self.max_position_size * self.portfolio_value / price
        ) * volatility_adj
        
        return max(0, position_size)
    
    def calculate_dynamic_stop_loss(self, 
                                  entry_price: float, 
                                  current_price: float,
                                  volatility: float,
                                  trend_strength: float) -> float:
        """Calculate dynamic stop loss based on volatility and trend."""
        
        # Base stop loss
        base_stop = entry_price * (1 - self.stop_loss_pct)
        
        # Volatility-adjusted stop loss
        volatility_stop = entry_price * (1 - self.stop_loss_pct * (1 + volatility * 5))
        
        # Trend-adjusted stop loss (tighter stops in weak trends)
        trend_adj = 1 + (1 - abs(trend_strength)) * 0.5
        trend_stop = entry_price * (1 - self.stop_loss_pct * trend_adj)
        
        # Use the most conservative stop loss
        stop_loss = max(base_stop, volatility_stop, trend_stop)
        
        return stop_loss
    
    def should_exit_position(self, 
                           position_id: str,
                           current_price: float,
                           current_volatility: float,
                           signal_strength: float) -> Tuple[bool, str]:
        """
        Determine if position should be exited based on risk management rules.
        
        Returns:
            (should_exit, reason)
        """
        
        if position_id not in self.positions:
            return False, "Position not found"
        
        position = self.positions[position_id]
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        shares = position['shares']
        
        # Check stop loss
        if current_price <= stop_loss:
            return True, "Stop loss triggered"
        
        # Check trailing stop
        if self.trailing_stop and 'highest_price' in position:
            trailing_stop = position['highest_price'] * (1 - self.stop_loss_pct * 0.5)
            if current_price <= trailing_stop:
                return True, "Trailing stop triggered"
        
        # Check signal reversal
        if position['signal_type'] == 'long' and signal_strength < -0.3:
            return True, "Signal reversal (long to short)"
        elif position['signal_type'] == 'short' and signal_strength > 0.3:
            return True, "Signal reversal (short to long)"
        
        # Check volatility-based exit
        if current_volatility > position['entry_volatility'] * 2:
            return True, "Excessive volatility"
        
        # Check time-based exit (max 30 days)
        days_held = (pd.Timestamp.now() - position['entry_time']).days
        if days_held > 30:
            return True, "Time-based exit"
        
        return False, "Hold position"
    
    def update_position(self, 
                       position_id: str, 
                       current_price: float,
                       current_volatility: float):
        """Update position tracking information."""
        
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # Update highest price for trailing stop
            if current_price > position.get('highest_price', 0):
                position['highest_price'] = current_price
            
            # Update current P&L
            if position['signal_type'] == 'long':
                position['current_pnl'] = (current_price - position['entry_price']) * position['shares']
            else:  # short
                position['current_pnl'] = (position['entry_price'] - current_price) * position['shares']
            
            # Update volatility
            position['current_volatility'] = current_volatility
    
    def execute_trade(self, 
                     signal: Dict,
                     current_price: float,
                     current_volatility: float,
                     trend_strength: float) -> Dict:
        """
        Execute a trade with risk management.
        
        Args:
            signal: Dictionary with 'action', 'confidence', 'signal_strength'
            current_price: Current asset price
            current_volatility: Current volatility estimate
            trend_strength: Current trend strength (-1 to 1)
        """
        
        action = signal['action']
        confidence = signal['confidence']
        signal_strength = signal['signal_strength']
        
        # Calculate position size
        position_size = self.calculate_position_size(
            signal_strength, current_volatility, current_price, confidence
        )
        
        if position_size == 0:
            return {'action': 'no_trade', 'reason': 'Insufficient position size'}
        
        # Calculate stop loss
        stop_loss = self.calculate_dynamic_stop_loss(
            current_price, current_price, current_volatility, trend_strength
        )
        
        # Create position
        position_id = f"pos_{len(self.positions) + 1}"
        position = {
            'entry_price': current_price,
            'shares': position_size,
            'stop_loss': stop_loss,
            'signal_type': action,
            'entry_time': pd.Timestamp.now(),
            'entry_volatility': current_volatility,
            'current_pnl': 0,
            'highest_price': current_price if action == 'long' else 0
        }
        
        self.positions[position_id] = position
        
        return {
            'action': action,
            'position_id': position_id,
            'shares': position_size,
            'stop_loss': stop_loss,
            'risk_amount': position_size * (current_price - stop_loss)
        }
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate current portfolio metrics."""
        
        total_pnl = sum(pos['current_pnl'] for pos in self.positions.values())
        total_risk = sum(pos['shares'] * (pos['entry_price'] - pos['stop_loss']) 
                        for pos in self.positions.values())
        
        return {
            'total_positions': len(self.positions),
            'total_pnl': total_pnl,
            'portfolio_value': self.portfolio_value + total_pnl,
            'total_risk': total_risk,
            'risk_pct': total_risk / self.portfolio_value,
            'positions': self.positions
        }

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced features for risk management."""
    
    df = df.copy()
    
    # Volatility features
    df['Volatility_5d'] = df['Close'].rolling(5).std() / df['Close'].rolling(5).mean()
    df['Volatility_20d'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
    df['Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
    
    # Trend strength
    df['Trend_Strength'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Market regime
    df['Market_Regime'] = np.where(df['Close'] > df['Close'].rolling(50).mean(), 'bull', 'bear')
    
    # Support/Resistance levels
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    df['Price_vs_Support'] = (df['Close'] - df['Support']) / df['Close']
    df['Price_vs_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
    
    # Volume analysis
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

def backtest_with_risk_management(df: pd.DataFrame, 
                                 predictions: List[float],
                                 initial_capital: float = 100000) -> Dict:
    """
    Backtest trading strategy with advanced risk management.
    
    Args:
        df: Price data DataFrame
        predictions: Model predictions (-1 to 1)
        initial_capital: Starting capital
    """
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    risk_manager.portfolio_value = initial_capital
    
    # Calculate advanced features
    df = calculate_advanced_features(df)
    
    trades = []
    portfolio_values = [initial_capital]
    
    for i in range(len(df)):
        if i < 20:  # Skip first 20 days for feature calculation
            continue
            
        current_price = df['Close'].iloc[i]
        current_volatility = df['Volatility_20d'].iloc[i]
        trend_strength = df['Trend_Strength'].iloc[i]
        prediction = predictions[i] if i < len(predictions) else 0
        
        # Check existing positions for exits
        positions_to_close = []
        for pos_id, position in risk_manager.positions.items():
            should_exit, reason = risk_manager.should_exit_position(
                pos_id, current_price, current_volatility, prediction
            )
            
            if should_exit:
                positions_to_close.append((pos_id, reason))
        
        # Close positions
        for pos_id, reason in positions_to_close:
            position = risk_manager.positions[pos_id]
            pnl = position['current_pnl']
            risk_manager.portfolio_value += pnl
            
            trades.append({
                'type': 'exit',
                'position_id': pos_id,
                'price': current_price,
                'pnl': pnl,
                'reason': reason,
                'date': df.index[i]
            })
            
            del risk_manager.positions[pos_id]
        
        # Update remaining positions
        for pos_id in risk_manager.positions:
            risk_manager.update_position(pos_id, current_price, current_volatility)
        
        # Generate new signals
        if abs(prediction) > 0.3:  # Only trade on strong signals
            signal = {
                'action': 'long' if prediction > 0 else 'short',
                'confidence': abs(prediction),
                'signal_strength': prediction
            }
            
            # Check if we can take new positions
            portfolio_metrics = risk_manager.get_portfolio_metrics()
            if portfolio_metrics['risk_pct'] < 0.1:  # Max 10% portfolio risk
                
                trade_result = risk_manager.execute_trade(
                    signal, current_price, current_volatility, trend_strength
                )
                
                if trade_result['action'] != 'no_trade':
                    trades.append({
                        'type': 'entry',
                        'position_id': trade_result['position_id'],
                        'price': current_price,
                        'shares': trade_result['shares'],
                        'stop_loss': trade_result['stop_loss'],
                        'date': df.index[i]
                    })
        
        # Update portfolio value
        current_pnl = sum(pos['current_pnl'] for pos in risk_manager.positions.values())
        portfolio_values.append(initial_capital + current_pnl)
    
    # Calculate final metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate Sharpe ratio
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }

if __name__ == "__main__":
    # Example usage
    print("Advanced Risk Management System")
    print("This module provides:")
    print("- Dynamic position sizing using Kelly Criterion")
    print("- Adaptive stop-loss based on volatility")
    print("- Portfolio-level risk management")
    print("- Multi-timeframe analysis")
    print("- Advanced exit strategies") 