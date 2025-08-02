# Jesse bulk backtest

A tool for bulk backtesting Jesse trading strategies, designed for the latest Jesse framework that uses Optuna for optimization.

## Features

`jesse-bulk pick`
Loads optimization results from Optuna SQLite database files. Removes duplicates and filters results according to your config.

`jesse-bulk refine`
Runs backtests with specific DNAs from Optuna optimization results.

`jesse-bulk bulk`
Runs all backtests according to your configuration (symbols, timeframes, start & finish-dates) without specific DNAs.

## Requirements

- Latest Jesse framework with Optuna optimization support
- Jesse project with completed optimization runs (Optuna database files)

You will find the results in a csv in your project folder. 

Uses joblib for multiprocessing. Uses pickle cache for candles. You might want to clear `storage/bulk` if you use it a lot and run out of space.

The bulk_config.yml should be self-explanatory.

### Configuration

You can specify the Optuna study name in `bulk_config.yml` if multiple studies exist in the database:

```yaml
# Optional: specify Optuna study name (if not provided, uses most recent study)
optuna_study_name: 'MyStrategy_optuna_ray_session123'
```

If no study name is specified, jesse-bulk will automatically use the most recent study found in the database.

## Warning
- warm-up-candles are taken from the candles passed. So the actual start_date is different then it would be during a normal backtest.
- extra route candles are added to all backtests - even though they might not be needed by the symbol. 

This could be improved.


# Installation

```sh
# install from git
pip install git+https://github.com/btagliani/jesse-bulk.git

# cd in your Jesse project directory

# create the config file
jesse-bulk create-config

# edit the created yml file in your project directory 

# pick / filter optimization results from Optuna database
jesse-bulk pick storage/temp/optuna/optuna_study.db

# refine bulk backtests with DNAs from Optuna database
jesse-bulk refine StrategyName storage/temp/optuna/optuna_study.db

# bulk backtests (no specific DNAs)
jesse-bulk bulk StrategyName 

```


## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. Do not risk money which you are afraid to lose. There might be bugs in the code - this software DOES NOT come with ANY warranty.
