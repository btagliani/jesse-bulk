# Jesse Bulk Backtest

A powerful tool for bulk backtesting Jesse trading strategies, fully compatible with the latest Jesse framework that uses Optuna for optimization. Run thousands of backtests in parallel to validate your strategies across different parameters and time periods.

## 🚀 Features

### Core Commands

**`jesse-bulk pick [db_path]`**  
Loads optimization results from Optuna SQLite database files. Removes duplicates and filters results according to your configuration criteria.

**`jesse-bulk refine [strategy] [db_path]`**  
Runs backtests with specific DNAs from Optuna optimization results. Perfect for validating optimized parameters across your configured timeframes and symbols.

**`jesse-bulk bulk [strategy]`**  
Runs basic backtests according to your configuration (symbols, timeframes, timespans) without specific hyperparameters. Uses default strategy parameters.

**`jesse-bulk refine-best [db_path] --top-n N --runs-per-dna R`**  
🆕 Tests the top N performing DNAs with R random time periods each. Ideal for robustness testing of your best strategies. Supports multiple selection presets (conservative, aggressive, balanced, robust) or test all presets with `--all-presets`.

**`jesse-bulk hall-of-fame [strategy] --top-n N --runs-per-dna R`**  
🆕 Tests the best DNAs from your Hall of Fame (persistent storage of top performers). Awards wins based on composite performance scoring.

**`jesse-bulk leaderboard --min-wins 1`**  
🆕 Shows the Hall of Fame leaderboard ranked by total wins. DNAs earn wins by consistently performing well across different market conditions.

**`jesse-bulk prune-hall-of-fame --min-success-rate 80 --confirm`**  
🆕 Removes underperforming DNAs from Hall of Fame based on test results. Use --confirm to actually delete (otherwise dry run).

**`jesse-bulk create-config`**  
Creates a `bulk_config.yml` template file in your project directory.

### Key Capabilities

- **Parallel Processing**: Utilizes all CPU cores for maximum performance
- **Optuna Integration**: Native support for Optuna SQLite databases with base64-encoded DNAs
- **Multi-timeframe Support**: Correctly handles warmup candles for any timeframe (1m, 3h, 4h, 1d, etc.)
- **Smart Caching**: Uses Jesse's existing candle cache system for optimal performance
- **Filtering & Sorting**: Advanced filtering based on performance metrics
- **Result Analysis**: CSV output with comprehensive backtest metrics
- **Hall of Fame**: Persistent storage of best performing DNAs with win tracking
- **Selection Presets**: Multiple DNA selection strategies (conservative, aggressive, balanced, robust)

## 📋 Requirements

- **Jesse Framework**: Latest version with Optuna optimization support
- **Python**: 3.8+
- **Jesse Project**: With completed optimization runs (Optuna database files)
- **Strategies**: Your custom Jesse strategies with hyperparameters

## 📊 Performance

- **Speed**: Process 2,500+ backtests in under 2 minutes
- **Efficiency**: Automatic candle caching and parallel execution
- **Scalability**: Handles any number of symbols, timeframes, and hyperparameter combinations

Results are saved as CSV files in your project folder with comprehensive metrics including win rate, profit percentage, Sharpe ratio, and more.

**Storage Note**: Uses pickle cache for candles in `storage/bulk`. Consider clearing this directory periodically if you run many backtests.

## ⚙️ Configuration

Jesse-bulk uses a `bulk_config.yml` file for configuration. Generate it with `jesse-bulk create-config`.

### Key Configuration Options

```yaml
# Sorting criteria for optimization results
sort_by: "training_log.net_profit_percentage"

# CPU cores to use (-1 for all cores)
n_jobs: 16

# Optuna study name (optional - auto-detects if not specified)
optuna_study_name: 'MyStrategy_optuna_ray_session123'

# Backtest configuration
backtest-data:
  strategy_name: YourStrategy
  starting_balance: 25000
  fee: 0.0004
  futures_leverage: 10
  warm_up_candles: 210  # Strategy-specific warmup period
  
  # Trading pairs and timeframes
  symbols:
    - "BTC-USDT"
    - "ETH-USDT"
  timeframes:
    - "3h"
    - "4h"
    
  # Time periods for backtesting
  timespans:
    0:
      start_date: "2024-01-01"
      finish_date: "2024-12-31"

# Advanced filtering (optional)
filter_dna:
  training:
    win_rate:
      min: 0.6  # Minimum 60% win rate
    net_profit_percentage:
      min: 100  # Minimum 100% profit
```

### Auto-Detection Features

- **Study Detection**: Automatically finds the most recent Optuna study if no `optuna_study_name` is specified
- **Database Discovery**: Auto-locates Optuna databases in your project if path not provided
- **Timeframe Warmup**: Automatically calculates correct warmup periods for different timeframes

## ⚠️ Important Notes

- **Warmup Handling**: The tool correctly calculates warmup candles for your strategy's timeframe (e.g., 210 × 180 minutes = 37,800 1-minute candles for 3h timeframe)
- **Extra Routes**: Extra route candles are added to all backtests when configured, even if not needed by all symbols
- **Date Ranges**: For `refine-best`, random start dates are generated within the specified constraints

# 📦 Installation

```bash
# Install from git
pip install git+https://github.com/btagliani/jesse-bulk.git

# Navigate to your Jesse project directory
cd /path/to/your/jesse-project

# Create the configuration file
jesse-bulk create-config

# Edit the created bulk_config.yml file to match your strategy and preferences
```

## 🚀 Quick Start

### 1. Basic Workflow

```bash
# Step 1: Create and configure bulk_config.yml
jesse-bulk create-config

# Step 2: Filter optimization results (optional)
jesse-bulk pick storage/temp/optuna/optuna_study.db

# Step 3: Run bulk backtests with optimized parameters
jesse-bulk refine YourStrategy storage/temp/optuna/optuna_study.db

# Step 4: Test top performers with random time periods
jesse-bulk refine-best storage/temp/optuna/optuna_study.db --top-n 10 --runs-per-dna 5
```

### 2. Advanced Workflow: refine-best → hall-of-fame → leaderboard

This workflow demonstrates the full DNA selection and tracking system:

```bash
# Step 1: Find the best DNAs using advanced selection presets
# Test all presets to find the "king DNA"
jesse-bulk refine-best storage/temp/optuna/optuna_study.db --all-presets --top-n 5 --runs-per-dna 10

# Output: Finds top performers across conservative, aggressive, balanced, and robust selection methods
# The best DNAs are automatically added to the Hall of Fame

# Step 2: Test Hall of Fame DNAs across different market conditions
# This awards "wins" to DNAs that consistently perform well
jesse-bulk hall-of-fame --top-n 10 --runs-per-dna 10 --min-trades 20

# Output: Tests stored DNAs and awards wins based on composite scoring:
# - 50% Success Rate (avoiding bankruptcy is critical)
# - 20% Average Profit
# - 15% Average Sharpe Ratio
# - 10% Average Finishing Balance
# - 5% Minimum Balance

# Step 3: View the leaderboard to see which DNAs consistently win
jesse-bulk leaderboard

# Output shows:
# 🏆 HALL OF FAME LEADERBOARD
# Rank  DNA          Wins   Avg Score  Total Score  Strategy
# 1     ...0725b7a2  3      85.2       255.6        Rainmaker
# 2     ...90a748d0  2      78.5       157.0        Rainmaker

# Step 4: Clean up underperformers (optional)
# Remove DNAs that don't meet performance criteria
jesse-bulk prune-hall-of-fame --min-success-rate 80 --min-avg-profit 200 --confirm
```

### Selection Presets Explained

When using `refine-best`, you can choose different selection strategies:

- **`--selection-preset conservative`**: Prioritizes consistency and low drawdown
- **`--selection-preset aggressive`**: Focuses on maximum returns
- **`--selection-preset balanced`**: Balances risk and reward (default)
- **`--selection-preset robust`**: Emphasizes cross-validation and overfitting detection
- **`--all-presets`**: Tests all presets to find the ultimate best DNA

### 2. Command Examples

```bash
# Run basic backtests without specific DNAs
jesse-bulk bulk MyStrategy

# Test top 5 DNAs with 3 random time periods each
jesse-bulk refine-best /path/to/study.db --top-n 5 --runs-per-dna 3

# Auto-detect database location (if only one exists)
jesse-bulk refine MyStrategy auto

# Process specific Optuna study
jesse-bulk pick /path/to/optuna_study.db
```

## 📈 Example Results

After running `jesse-bulk refine`, you'll get comprehensive CSV results:

```
Strategy: Rainmaker
Total Backtests: 2,500
Execution Time: 1.3 minutes
Best DNA: 734.14% finishing balance
```

The CSV output includes metrics like:
- `total` - Number of trades
- `win_rate` - Winning percentage
- `net_profit_percentage` - Total profit/loss %
- `sharpe_ratio` - Risk-adjusted returns
- `max_drawdown` - Maximum loss period
- And 30+ other detailed metrics

## 🔧 Advanced Usage

### Custom Filtering

Filter results based on multiple criteria in `bulk_config.yml`:

```yaml
filter_dna:
  training:
    total:
      min: 50        # Minimum trades
    win_rate:
      min: 0.55      # Minimum 55% win rate
    sharpe_ratio:
      min: 1.0       # Minimum Sharpe ratio
    max_drawdown:
      max: 0.2       # Maximum 20% drawdown
```

### Multiple Timeframes and Symbols

```yaml
symbols:
  - "BTC-USDT"
  - "ETH-USDT"
  - "SOL-USDT"
timeframes:
  - "1h"
  - "3h"
  - "4h"
timespans:
  0:
    start_date: "2024-01-01"
    finish_date: "2024-06-30"
  1:
    start_date: "2024-07-01"  
    finish_date: "2024-12-31"
```

This configuration will run backtests for each combination: 3 symbols × 2 timeframes × 2 timespans = 12 backtests per DNA.

## 🏆 Hall of Fame System

The Hall of Fame is a persistent storage system that tracks your best performing DNAs over time. It solves a common problem: finding DNAs that perform well not just in their optimization period, but consistently across different market conditions.

### How It Works

1. **DNA Storage**: When you run `refine-best`, top performers are automatically added to the Hall of Fame SQLite database (`storage/hall_of_fame.db`)

2. **Performance Testing**: Use `hall-of-fame` command to test stored DNAs across random time periods

3. **Win System**: DNAs earn "wins" based on composite scoring:
   - **50%** - Success Rate (must be ≥80% to qualify)
   - **20%** - Average Profit
   - **15%** - Average Sharpe Ratio
   - **10%** - Average Finishing Balance
   - **5%** - Minimum Balance (risk control)

4. **Leaderboard**: Track which DNAs consistently win across different tests

### Why Success Rate Matters Most

The scoring heavily weights success rate (50%) because in real trading:
- A strategy that fails 60% of the time means bankruptcy 6 out of 10 times
- You can't compound gains if you go broke
- Consistency beats home runs

### Hall of Fame Commands

```bash
# View statistics
jesse-bulk hall-of-fame --show-stats

# Test top 10 DNAs with 5 runs each
jesse-bulk hall-of-fame --top-n 10 --runs-per-dna 5

# Export to CSV
jesse-bulk hall-of-fame --export-csv my_hall_of_fame.csv

# View leaderboard
jesse-bulk leaderboard

# Clean up underperformers
jesse-bulk prune-hall-of-fame --min-success-rate 80 --confirm
```

## 🐛 Troubleshooting

### Common Issues

**❌ "No Optuna database found"**
```bash
# Specify full path to your Optuna database
jesse-bulk refine MyStrategy /full/path/to/storage/temp/optuna/optuna_study.db
```

**❌ "IndexError: index -X is out of bounds"**  
✅ **Fixed in latest version!** The tool now correctly calculates warmup candles for all timeframes.

**❌ "Strategy not found"**  
Make sure you're in your Jesse project directory and the strategy name matches exactly.

**❌ Performance issues**  
Reduce `n_jobs` in config or clear the `storage/bulk` cache directory.

### Getting Help

- Check that you're in a valid Jesse project directory
- Ensure your Optuna optimization has completed successfully  
- Verify your strategy name matches the directory name in `strategies/`
- Make sure `bulk_config.yml` is properly configured

## 🔄 What's New

### v2.1 Updates (Latest)
- ✅ **Hall of Fame System**: Persistent storage of best performing DNAs
- ✅ **Win Tracking**: DNAs earn wins through consistent performance
- ✅ **Selection Presets**: Conservative, aggressive, balanced, and robust selection methods
- ✅ **Leaderboard**: Track which DNAs consistently outperform
- ✅ **Performance Pruning**: Remove underperforming DNAs based on test results
- ✅ **Composite Scoring**: Multi-criteria evaluation prioritizing success rate (50% weight)
- ✅ **All-Presets Mode**: Test all selection methods to find the ultimate "king DNA"

### v2.0 Updates
- ✅ **Full Optuna Support**: Native integration with Optuna SQLite databases
- ✅ **Base64 DNA Decoding**: Handles modern Jesse optimization format
- ✅ **Multi-timeframe Fix**: Correctly calculates warmup for any timeframe
- ✅ **New refine-best Command**: Test top performers with random periods
- ✅ **Performance Boost**: 2,500+ backtests in under 2 minutes
- ✅ **Auto-detection**: Smart database and study discovery
- ❌ **Removed**: Legacy CSV support (backwards compatibility removed)


## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. Do not risk money which you are afraid to lose. There might be bugs in the code - this software DOES NOT come with ANY warranty.
