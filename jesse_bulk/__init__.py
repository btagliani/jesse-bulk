import datetime
import logging
import os
import pathlib
import pickle
import random
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import click
import jesse.helpers as jh
import joblib
import numpy as np
import pandas as pd
import pkg_resources
import yaml
from jesse.research import get_candles
from jesse.research.backtest import backtest_with_dates


def start_logger_if_necessary():
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.ERROR)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s %(key)s %(message)s'))
        fh = logging.FileHandler('bulk.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s %(key)s %(message)s'))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger

# create a Click group
@click.group()
@click.version_option(pkg_resources.get_distribution("jesse-bulk").version)
def cli() -> None:
    pass


@cli.command()
def create_config() -> None:
    validate_cwd()
    target_dirname = pathlib.Path().resolve()
    package_dir = pathlib.Path(__file__).resolve().parent
    shutil.copy2(f'{package_dir}/bulk_config.yml', f'{target_dirname}/bulk_config.yml')


@cli.command()
@click.argument('csv_path', required=True, type=str)
def pick(csv_path: str) -> None:
    from .picker import filter_and_sort_dna_df

    cfg = get_config()

    filter_and_sort_dna_df(csv_path, cfg)


@cli.command()
@click.argument('strategy_name', required=True, type=str)
@click.argument('csv_path', required=True, type=str)
def refine(strategy_name: str, csv_path: str) -> None:
    validate_cwd()

    dna_df = pd.read_csv(csv_path, encoding='utf-8', sep='\t')

    # old csv
    if 'dna' not in dna_df:
        dna_df = pd.read_csv(csv_path, encoding='utf-8')
        dna_df.rename(columns={"training_log.win-rate": "training_log.win_rate",
                               "training_log.PNL": "training_log.net_profit_percentage",
                               "testing_log.win-rate": "testing_log.win_rate",
                               "testing_log.PNL": "testing_log.net_profit_percentage"}, inplace=True)
    cfg = get_config()

    StrategyClass = jh.get_strategy_class(strategy_name)
    hp_dict = StrategyClass().hyperparameters()

    config = {
        'starting_balance': cfg['backtest-data']['starting_balance'],
        'fee': cfg['backtest-data']['fee'],
        'type': cfg['backtest-data']['type'],
        'futures_leverage': cfg['backtest-data']['futures_leverage'],
        'futures_leverage_mode': cfg['backtest-data']['futures_leverage_mode'],
        'exchange': cfg['backtest-data']['exchange'],
        'settlement_currency': cfg['backtest-data']['settlement_currency'],
        'warm_up_candles': cfg['backtest-data']['warm_up_candles']
    }

    if len(cfg['backtest-data']['symbols']) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg['backtest-data']['timeframes']) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg['backtest-data']['timespans']) == 0:
        raise ValueError("You need to define a timespan. Check your config.")

    mp_args = []
    for symbol in cfg['backtest-data']['symbols']:
        for timeframe in cfg['backtest-data']['timeframes']:
            for timespan in cfg['backtest-data']['timespans'].items():
                timespan = timespan[1]
                candles = {}
                extra_routes = []
                if len(cfg['backtest-data']['extra_routes']) != 0:
                    for extra_route in cfg['backtest-data']['extra_routes'].items():
                        extra_route = extra_route[1]
                        candles[jh.key(extra_route['exchange'], extra_route['symbol'])] = {
                            'exchange': extra_route['exchange'],
                            'symbol': extra_route['symbol'],
                            'candles': get_candles_with_cache(
                                extra_route['exchange'],
                                extra_route['symbol'],
                                timespan['start_date'],
                                timespan['finish_date'],
                            ),
                        }
                        extra_routes.append({'exchange': extra_route['exchange'], 'symbol': extra_route['symbol'],
                                             'timeframe': extra_route['timeframe']})
                candles[jh.key(cfg['backtest-data']['exchange'], symbol)] = {
                    'exchange': cfg['backtest-data']['exchange'],
                    'symbol': symbol,
                    'candles': get_candles_with_cache(
                        cfg['backtest-data']['exchange'],
                        symbol,
                        timespan['start_date'],
                        timespan['finish_date'],
                    ),
                }

                route = [{'exchange': cfg['backtest-data']['exchange'], 'strategy': strategy_name, 'symbol': symbol,
                          'timeframe': timeframe}]

                for dna in dna_df['dna']:
                    key = f'{symbol}-{timeframe}-{timespan["start_date"]}-{timespan["finish_date"]}-{dna}'
                    mp_args.append((key, config, route, extra_routes, timespan['start_date'], timespan['finish_date'], hp_dict, dna))

    n_jobs = joblib.cpu_count() if cfg['n_jobs'] == -1 else cfg['n_jobs']

    print('Starting bulk backtest.')
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    results = parallel(
        joblib.delayed(backtest_with_info_key)(*args)
        for args in mp_args
    )

    old_name = pathlib.Path(csv_path).stem
    new_path = pathlib.Path(csv_path).with_stem(f'{old_name}-results')

    results_df = pd.DataFrame.from_dict(results, orient='columns')

    results_df.to_csv(new_path, header=True, index=False, encoding='utf-8', sep='\t')


@cli.command()
@click.argument('strategy_name', required=True, type=str)
def bulk(strategy_name: str) -> None:
    validate_cwd()
    cfg = get_config()

    config = {
        'starting_balance': cfg['backtest-data']['starting_balance'],
        'fee': cfg['backtest-data']['fee'],
        'type': cfg['backtest-data']['type'],
        'futures_leverage': cfg['backtest-data']['futures_leverage'],
        'futures_leverage_mode': cfg['backtest-data']['futures_leverage_mode'],
        'exchange': cfg['backtest-data']['exchange'],
        'settlement_currency': cfg['backtest-data']['settlement_currency'],
        'warm_up_candles': cfg['backtest-data']['warm_up_candles']
    }

    if len(cfg['backtest-data']['symbols']) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg['backtest-data']['timeframes']) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg['backtest-data']['timespans']) == 0:
        raise ValueError("You need to define a timespan. Check your config.")

    mp_args = []
    for symbol in cfg['backtest-data']['symbols']:
        for timeframe in cfg['backtest-data']['timeframes']:
            for timespan in cfg['backtest-data']['timespans'].items():
                timespan = timespan[1]
                candles = {}
                extra_routes = []
                if len(cfg['backtest-data']['extra_routes']) != 0:
                    for extra_route in cfg['backtest-data']['extra_routes'].items():
                        extra_route = extra_route[1]
                        candles[jh.key(extra_route['exchange'], extra_route['symbol'])] = {
                            'exchange': extra_route['exchange'],
                            'symbol': extra_route['symbol'],
                            'candles': get_candles_with_cache(
                                extra_route['exchange'],
                                extra_route['symbol'],
                                timespan['start_date'],
                                timespan['finish_date'],
                            ),
                        }
                        extra_routes.append({'exchange': extra_route['exchange'], 'symbol': extra_route['symbol'],
                                             'timeframe': extra_route['timeframe']})
                candles[jh.key(cfg['backtest-data']['exchange'], symbol)] = {
                    'exchange': cfg['backtest-data']['exchange'],
                    'symbol': symbol,
                    'candles': get_candles_with_cache(
                        cfg['backtest-data']['exchange'],
                        symbol,
                        timespan['start_date'],
                        timespan['finish_date'],
                    ),
                }

                route = [{'exchange': cfg['backtest-data']['exchange'], 'strategy': strategy_name, 'symbol': symbol,
                          'timeframe': timeframe}]

                key = f'{symbol}-{timeframe}-{timespan["start_date"]}-{timespan["finish_date"]}'

                mp_args.append((key, config, route, extra_routes, timespan['start_date'], timespan['finish_date'], None, None))

    n_jobs = joblib.cpu_count() if cfg['n_jobs'] == -1 else cfg['n_jobs']

    print('Starting bulk backtest.')
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    results = parallel(
        joblib.delayed(backtest_with_info_key)(*args)
        for args in mp_args
    )

    results_df = pd.DataFrame.from_dict(results, orient='columns')

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")

    results_df.to_csv(f'{strategy_name}_bulk_{dt}.csv', header=True, index=False, encoding='utf-8', sep='\t')

@cli.command()
@click.argument('csv_path', required=True, type=str)
def refine_best(csv_path: str) -> None:
    validate_cwd()

    cfg = get_config()
    strategy_name = cfg['backtest-data']['strategy_name']
    dna_df = pd.read_csv(csv_path, header=None, sep=';')[0].to_frame(name='dna')

    StrategyClass = jh.get_strategy_class(strategy_name)
    hp_dict = StrategyClass().hyperparameters()

    print(dna_df)  # Now this should print the correct DNA values

    all_results = []
    for _ in range(10):  # Run 10 backtests for each DNA
        config, mp_args = prepare_backtest_config_and_args(strategy_name, cfg, hp_dict, dna_df['dna'].tolist())
        results = run_parallel_backtest(mp_args, cfg)
        all_results.extend(results)

    # Convert all results to a DataFrame
    results_df = pd.DataFrame.from_dict(all_results, orient='columns')

    # Add a column for the DNA based on the 'key' column
    results_df['dna'] = results_df['key'].apply(lambda x: x.split('-')[-1])

    # Group by DNA and calculate the average for each group
    average_results = results_df.groupby('dna').mean()

    # Sort the groups based on the average of the selected metric
    sorted_results = average_results.sort_values(by='finishing_balance', ascending=False)

    # Save the results to a CSV file
    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")
    sorted_results.to_csv(f'{strategy_name}_refined_best_average_{dt}.csv', encoding='utf-8', sep='\t')

    # Print the best performing DNA
    best_dna = sorted_results.iloc[0]
    print(f"The best performing DNA is: {best_dna.name} with an average finishing balance of {best_dna['finishing_balance']:.2f}%")

def prepare_backtest_config_and_args(strategy_name: str, cfg: Dict, hp_dict: Dict, dnas: List[str]) -> Tuple[Dict, List[Tuple]]:
    config = {
        'starting_balance': cfg['backtest-data']['starting_balance'],
        'fee': cfg['backtest-data']['fee'],
        'type': cfg['backtest-data']['type'],
        'futures_leverage': cfg['backtest-data']['futures_leverage'],
        'futures_leverage_mode': cfg['backtest-data']['futures_leverage_mode'],
        'exchange': cfg['backtest-data']['exchange'],
        'settlement_currency': cfg['backtest-data']['settlement_currency'],
        'warm_up_candles': cfg['backtest-data']['warm_up_candles']
    }

    validate_backtest_data_config(cfg)

    warm_up_days = cfg['backtest-data']['warm_up_candles']  # Assuming warm-up is defined in candles (days)
    end_date = cfg['backtest-data']['end_date']
    mp_args = []
    for dna in dnas:
        start_date = get_random_dates_within_timespan(end_date, warm_up_days)
        for symbol in cfg['backtest-data']['symbols']:
            for timeframe in cfg['backtest-data']['timeframes']:
                route = [{'exchange': cfg['backtest-data']['exchange'], 'strategy': strategy_name, 'symbol': symbol, 'timeframe': timeframe}]
                key = f'{symbol}-{timeframe}-{start_date}-{end_date}-{dna}'
                extra_routes = []  # Assume you have a way to set extra_routes
                mp_args.append((key, config, route, extra_routes, start_date, end_date, hp_dict, dna))
    return config, mp_args

def get_random_dates_within_timespan(end_date: str, warm_up_days: int) -> str:
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Ensure there are enough days in the year for both the warm-up period and the backtest
    latest_start_day = end_date - timedelta(days=max(warm_up_days, 1) + 1)  # Ensure at least one day for backtesting

    # Generate a random start date
    start_date = latest_start_day - timedelta(days=random.randint(60, (latest_start_day - datetime(latest_start_day.year, 1, 1)).days))

    # Deduct the warm-up period
    start_date -= timedelta(days=warm_up_days)

    # Format dates to string in 'YYYY-MM-DD' format
    start_date_str = start_date.strftime('%Y-%m-%d')

    return start_date_str

def run_parallel_backtest(mp_args: List[Tuple], cfg: Dict) -> List[Dict]:
    n_jobs = joblib.cpu_count() if cfg['n_jobs'] == -1 else cfg['n_jobs']
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    results = parallel(joblib.delayed(backtest_with_info_key)(*args) for args in mp_args)
    return results

def validate_backtest_data_config(cfg: Dict) -> None:
    if len(cfg['backtest-data']['symbols']) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg['backtest-data']['timeframes']) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg['backtest-data']['timespans']) == 0:
        raise ValueError("You need to define a timespan. Check your config.")

def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    ls = os.listdir('.')
    is_jesse_project = 'strategies' in ls and 'storage' in ls

    if not is_jesse_project:
        print('Current directory is not a Jesse project. You must run commands from the root of a Jesse project.')
        exit()


def get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    path = pathlib.Path('storage/bulk')
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f'storage/bulk/{cache_file_name}')

    if cache_file.is_file():
        with open(f'storage/bulk/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        candles = get_candles(exchange, symbol, '1m', start_date, finish_date)
        with open(f'storage/bulk/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles


def backtest_with_info_key(key, config, route, extra_routes, start_date, end_date, hp_dict, dna):
    hp = jh.dna_to_hp(hp_dict, dna) if dna else None

    got_exception = False

    try:
        backtest_data = backtest_with_dates(config, route, extra_routes, start_date, end_date, hyperparameters = hp)['metrics']
    except Exception as e:
        logger = start_logger_if_necessary()
        logger.error("".join(traceback.TracebackException.from_exception(e).format()), extra={'key': key})
        # Re-raise the original exception so the Pool worker can
        # clean up
        got_exception = True

    if got_exception or backtest_data['total'] == 0:
        backtest_data = {'total': 0, 'total_winning_trades': None, 'total_losing_trades': None,
                         'starting_balance': None, 'finishing_balance': None, 'win_rate': None,
                         'ratio_avg_win_loss': None, 'longs_count': None, 'longs_percentage': None,
                         'shorts_percentage': None, 'shorts_count': None, 'fee': None, 'net_profit': None,
                         'net_profit_percentage': None, 'average_win': None, 'average_loss': None, 'expectancy': None,
                         'expectancy_percentage': None, 'expected_net_profit_every_100_trades': None,
                         'average_holding_period': None, 'average_winning_holding_period': None,
                         'average_losing_holding_period': None, 'gross_profit': None, 'gross_loss': None,
                         'max_drawdown': None, 'annual_return': None, 'sharpe_ratio': None, 'calmar_ratio': None,
                         'sortino_ratio': None, 'omega_ratio': None, 'serenity_index': None, 'smart_sharpe': None,
                         'smart_sortino': None, 'total_open_trades': None, 'open_pl': None, 'winning_streak': None,
                         'losing_streak': None, 'largest_losing_trade': None, 'largest_winning_trade': None,
                         'current_streak': None}
        if got_exception:
            backtest_data['total'] = "error"

    return {**{'key': key}, **backtest_data}


def get_config():
    cfg_file = pathlib.Path('bulk_config.yml')

    if not cfg_file.is_file():
        print("bulk_config not found. Run create-config command.")
        exit()
    else:
        with open("bulk_config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

    return cfg
