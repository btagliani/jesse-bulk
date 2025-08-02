import warnings
# Suppress pkg_resources deprecation warnings from Jesse (must be before Jesse imports)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

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
from jesse.research import get_candles, backtest


def start_logger_if_necessary():
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.ERROR)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(key)s %(message)s"))
        fh = logging.FileHandler("bulk.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(key)s %(message)s"))
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
    shutil.copy2(f"{package_dir}/bulk_config.yml", f"{target_dirname}/bulk_config.yml")


@cli.command()
@click.argument("db_path", required=True, type=str)
def pick(db_path: str) -> None:
    from .picker import filter_and_sort_dna_df

    cfg = get_config()

    filter_and_sort_dna_df(db_path, cfg)


@cli.command()
@click.argument("strategy_name", required=True, type=str)
@click.argument("db_path", required=True, type=str)
def refine(strategy_name: str, db_path: str) -> None:
    from .optuna_reader import read_optuna_study, find_optuna_databases
    
    validate_cwd()
    cfg = get_config()

    # Load data from Optuna database
    if not db_path.endswith('.db'):
        # Try to auto-detect Optuna database in project
        db_paths = find_optuna_databases()
        if db_paths:
            print(f"Found Optuna database: {db_paths[0]}")
            db_path = db_paths[0]
        else:
            raise ValueError("No Optuna database found. Please provide path to .db file")
    
    study_name = cfg.get('optuna_study_name', None)
    dna_df = read_optuna_study(db_path, study_name)

    StrategyClass = jh.get_strategy_class(strategy_name)
    hp_dict = StrategyClass().hyperparameters()

    config = {
        "starting_balance": cfg["backtest-data"]["starting_balance"],
        "fee": cfg["backtest-data"]["fee"],
        "type": cfg["backtest-data"]["type"],
        "futures_leverage": cfg["backtest-data"]["futures_leverage"],
        "futures_leverage_mode": cfg["backtest-data"]["futures_leverage_mode"],
        "exchange": cfg["backtest-data"]["exchange"],
        "settlement_currency": cfg["backtest-data"]["settlement_currency"],
        "warm_up_candles": cfg["backtest-data"]["warm_up_candles"],
    }

    if len(cfg["backtest-data"]["symbols"]) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg["backtest-data"]["timeframes"]) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg["backtest-data"]["timespans"]) == 0:
        raise ValueError("You need to define a timespan. Check your config.")

    mp_args = []
    for symbol in cfg["backtest-data"]["symbols"]:
        for timeframe in cfg["backtest-data"]["timeframes"]:
            for timespan in cfg["backtest-data"]["timespans"].items():
                timespan = timespan[1]
                candles = {}
                extra_routes = []
                if len(cfg["backtest-data"]["extra_routes"]) != 0:
                    for extra_route in cfg["backtest-data"]["extra_routes"].items():
                        extra_route = extra_route[1]
                        candles[
                            jh.key(extra_route["exchange"], extra_route["symbol"])
                        ] = {
                            "exchange": extra_route["exchange"],
                            "symbol": extra_route["symbol"],
                            "candles": get_candles_with_cache(
                                extra_route["exchange"],
                                extra_route["symbol"],
                                timespan["start_date"],
                                timespan["finish_date"],
                            ),
                        }
                        extra_routes.append(
                            {
                                "exchange": extra_route["exchange"],
                                "symbol": extra_route["symbol"],
                                "timeframe": extra_route["timeframe"],
                            }
                        )
                candles[jh.key(cfg["backtest-data"]["exchange"], symbol)] = {
                    "exchange": cfg["backtest-data"]["exchange"],
                    "symbol": symbol,
                    "candles": get_candles_with_cache(
                        cfg["backtest-data"]["exchange"],
                        symbol,
                        timespan["start_date"],
                        timespan["finish_date"],
                    ),
                }

                route = [
                    {
                        "exchange": cfg["backtest-data"]["exchange"],
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                ]

                for dna in dna_df["dna"]:
                    key = f'{symbol}-{timeframe}-{timespan["start_date"]}-{timespan["finish_date"]}-{dna}'
                    mp_args.append(
                        (
                            key,
                            config,
                            route,
                            extra_routes,
                            timespan["start_date"],
                            timespan["finish_date"],
                            hp_dict,
                            dna,
                        )
                    )

    n_jobs = joblib.cpu_count() if cfg["n_jobs"] == -1 else cfg["n_jobs"]

    print("Starting bulk backtest.")
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    results = parallel(
        joblib.delayed(backtest_with_info_key)(*args) for args in mp_args
    )

    old_name = pathlib.Path(db_path).stem
    new_path = pathlib.Path(db_path).with_suffix('.csv').with_stem(f"{old_name}-results")

    results_df = pd.DataFrame.from_dict(results, orient="columns")

    results_df.sort_values(by="finishing_balance", ascending=False, inplace=True)

    results_df.to_csv(new_path, header=True, index=False, encoding="utf-8", sep="\t")


@cli.command()
@click.argument("strategy_name", required=True, type=str)
def bulk(strategy_name: str) -> None:
    validate_cwd()
    cfg = get_config()

    config = {
        "starting_balance": cfg["backtest-data"]["starting_balance"],
        "fee": cfg["backtest-data"]["fee"],
        "type": cfg["backtest-data"]["type"],
        "futures_leverage": cfg["backtest-data"]["futures_leverage"],
        "futures_leverage_mode": cfg["backtest-data"]["futures_leverage_mode"],
        "exchange": cfg["backtest-data"]["exchange"],
        "settlement_currency": cfg["backtest-data"]["settlement_currency"],
        "warm_up_candles": cfg["backtest-data"]["warm_up_candles"],
    }

    if len(cfg["backtest-data"]["symbols"]) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg["backtest-data"]["timeframes"]) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg["backtest-data"]["timespans"]) == 0:
        raise ValueError("You need to define a timespan. Check your config.")

    mp_args = []
    for symbol in cfg["backtest-data"]["symbols"]:
        for timeframe in cfg["backtest-data"]["timeframes"]:
            for timespan in cfg["backtest-data"]["timespans"].items():
                timespan = timespan[1]
                extra_routes = []
                if len(cfg["backtest-data"]["extra_routes"]) != 0:
                    for extra_route in cfg["backtest-data"]["extra_routes"].items():
                        extra_route = extra_route[1]
                        extra_routes.append(
                            {
                                "exchange": extra_route["exchange"],
                                "symbol": extra_route["symbol"],
                                "timeframe": extra_route["timeframe"],
                            }
                        )

                route = [
                    {
                        "exchange": cfg["backtest-data"]["exchange"],
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                ]

                key = f'{symbol}-{timeframe}-{timespan["start_date"]}-{timespan["finish_date"]}'

                mp_args.append(
                    (
                        key,
                        config,
                        route,
                        extra_routes,
                        timespan["start_date"],
                        timespan["finish_date"],
                        None,  # hp_dict - not used for bulk
                        None,  # dna - not used for bulk
                    )
                )

    n_jobs = joblib.cpu_count() if cfg["n_jobs"] == -1 else cfg["n_jobs"]

    print("Starting bulk backtest.")
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    results = parallel(
        joblib.delayed(backtest_with_info_key)(*args) for args in mp_args
    )

    results_df = pd.DataFrame.from_dict(results, orient="columns")

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")

    results_df.to_csv(
        f"{strategy_name}_bulk_{dt}.csv",
        header=True,
        index=False,
        encoding="utf-8",
        sep="\t",
    )


@cli.command()
@click.argument("db_path", required=True, type=str)
@click.option("--top-n", default=10, help="Number of top DNAs to test (default: 10)")
@click.option("--runs-per-dna", default=10, help="Number of runs per DNA (default: 10)")
def refine_best(db_path: str, top_n: int, runs_per_dna: int) -> None:
    from .optuna_reader import read_optuna_study, find_optuna_databases
    
    validate_cwd()
    cfg = get_config()
    strategy_name = cfg["backtest-data"]["strategy_name"]

    # Load data from Optuna database
    if not db_path.endswith('.db'):
        # Try to auto-detect Optuna database in project
        db_paths = find_optuna_databases()
        if db_paths:
            print(f"Found Optuna database: {db_paths[0]}")
            db_path = db_paths[0]
        else:
            raise ValueError("No Optuna database found. Please provide path to .db file")
    
    study_name = cfg.get('optuna_study_name', None)
    dna_df = read_optuna_study(db_path, study_name)
    
    # Get top N DNAs based on the sort criteria from config
    sort_criteria = cfg.get('sort_by', 'training_log.net_profit_percentage')
    if 'training_log.' in sort_criteria:
        sort_column = sort_criteria.replace('training_log.', '')
    else:
        sort_column = sort_criteria
        
    # Sort and get top N
    if sort_column in dna_df.columns:
        top_dnas = dna_df.nlargest(top_n, sort_column)
    else:
        print(f"Warning: Sort column '{sort_column}' not found, using first {top_n} DNAs")
        top_dnas = dna_df.head(top_n)
    
    print(f"Selected top {len(top_dnas)} DNAs for refinement")

    StrategyClass = jh.get_strategy_class(strategy_name)
    hp_dict = StrategyClass().hyperparameters()

    # Prepare config (same as refine command)
    config = {
        "starting_balance": cfg["backtest-data"]["starting_balance"],
        "fee": cfg["backtest-data"]["fee"],
        "type": cfg["backtest-data"]["type"],
        "futures_leverage": cfg["backtest-data"]["futures_leverage"],
        "futures_leverage_mode": cfg["backtest-data"]["futures_leverage_mode"],
        "exchange": cfg["backtest-data"]["exchange"],
        "settlement_currency": cfg["backtest-data"]["settlement_currency"],
        "warm_up_candles": cfg["backtest-data"]["warm_up_candles"],
    }

    # Prepare arguments for multiple runs (using same logic as refine but with random dates)
    mp_args = []
    for _ in range(runs_per_dna):  # Multiple random time periods
        for symbol in cfg["backtest-data"]["symbols"]:
            for timeframe in cfg["backtest-data"]["timeframes"]:
                # Generate random dates for this run
                warm_up_days = cfg["backtest-data"]["warm_up_candles"]
                end_date = cfg["backtest-data"].get("end_date", "2025-07-30")
                start_date = get_random_dates_within_timespan(end_date, warm_up_days)
                
                # Prepare extra routes
                extra_routes = []
                if len(cfg["backtest-data"]["extra_routes"]) != 0:
                    for extra_route in cfg["backtest-data"]["extra_routes"].items():
                        extra_route = extra_route[1]
                        extra_routes.append({
                            "exchange": extra_route["exchange"],
                            "symbol": extra_route["symbol"],
                            "timeframe": extra_route["timeframe"],
                        })

                route = [{
                    "exchange": cfg["backtest-data"]["exchange"],
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                }]

                # Add one backtest for each DNA
                for dna in top_dnas["dna"]:
                    key = f'{symbol}-{timeframe}-{start_date}-{end_date}-{dna}'
                    mp_args.append((
                        key,
                        config,
                        route,
                        extra_routes,
                        start_date,
                        end_date,
                        hp_dict,
                        dna,
                    ))

    # Run all backtests in parallel
    n_jobs = joblib.cpu_count() if cfg["n_jobs"] == -1 else cfg["n_jobs"]
    print(f"Starting {len(mp_args)} backtests for top {len(top_dnas)} DNAs...")
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    all_results = parallel(
        joblib.delayed(backtest_with_info_key)(*args) for args in mp_args
    )

    # Convert all results to a DataFrame
    results_df = pd.DataFrame.from_dict(all_results, orient="columns")

    # Add a column for the DNA based on the 'key' column
    results_df["dna"] = results_df["key"].apply(lambda x: x.split("-")[-1])

    # Group by DNA and calculate the average for each group (only numeric columns)
    average_results = results_df.groupby("dna").mean(numeric_only=True)

    # Sort the groups based on the average of the selected metric
    sorted_results = average_results.sort_values(
        by="finishing_balance", ascending=False
    )

    # Save the results to a CSV file
    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")
    sorted_results.to_csv(
        f"{strategy_name}_refined_best_average_{dt}.csv", encoding="utf-8", sep="\t"
    )

    # Print the best performing DNA
    best_dna = sorted_results.iloc[0]
    print(
        f"The best performing DNA is: {best_dna.name} with an average finishing balance of {best_dna['finishing_balance']:.2f}%"
    )


def get_random_dates_within_timespan(end_date: str, warm_up_days: int) -> str:
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the latest possible start date (must leave room for warmup + at least 30 days of backtesting)
    latest_start_day = end_date - timedelta(days=30)  # 30 days of backtesting minimum
    
    # Calculate the earliest possible start date (go back further for more variety)
    max_lookback = max(120, warm_up_days + 60)  # At least 120 days back, or warmup + 60 days
    earliest_start_day = end_date - timedelta(days=max_lookback)

    # Ensure we have a valid range
    if latest_start_day <= earliest_start_day:
        # If range is invalid, just use a simple approach
        start_date = end_date - timedelta(days=60)  # Go back 60 days
    else:
        # Generate a random start date between earliest_start_day and latest_start_day
        total_days = (latest_start_day - earliest_start_day).days
        random_days = random.randint(0, total_days)
        start_date = earliest_start_day + timedelta(days=random_days)

    # Format dates to string in 'YYYY-MM-DD' format
    start_date_str = start_date.strftime("%Y-%m-%d")
    print(f"Start date: {start_date_str}")
    return start_date_str


def validate_backtest_data_config(cfg: Dict) -> None:
    if len(cfg["backtest-data"]["symbols"]) == 0:
        raise ValueError("You need to define a symbol. Check your config.")
    if len(cfg["backtest-data"]["timeframes"]) == 0:
        raise ValueError("You need to define a timeframe. Check your config.")
    if len(cfg["backtest-data"]["timespans"]) == 0:
        raise ValueError("You need to define a timespan. Check your config.")


def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    ls = os.listdir(".")
    is_jesse_project = "strategies" in ls and "storage" in ls

    if not is_jesse_project:
        print(
            "Current directory is not a Jesse project. You must run commands from the root of a Jesse project."
        )
        exit()


def get_candles_and_extra_routes(
    cfg: Dict, symbol: str, timespan: Dict
) -> Tuple[Dict, List[Dict]]:
    candles = {}
    extra_routes = []

    # Fetch candles for main trading pair
    candles[jh.key(cfg["backtest-data"]["exchange"], symbol)] = {
        "exchange": cfg["backtest-data"]["exchange"],
        "symbol": symbol,
        "candles": get_candles_with_cache(
            cfg["backtest-data"]["exchange"],
            symbol,
            timespan["start_date"],
            timespan["finish_date"],
        ),
    }

    # Fetch candles for extra routes if defined
    if cfg["backtest-data"].get("extra_routes"):
        for extra_route in cfg["backtest-data"]["extra_routes"]:
            candles[jh.key(extra_route["exchange"], extra_route["symbol"])] = {
                "exchange": extra_route["exchange"],
                "symbol": extra_route["symbol"],
                "candles": get_candles_with_cache(
                    extra_route["exchange"],
                    extra_route["symbol"],
                    timespan["start_date"],
                    timespan["finish_date"],
                ),
            }
            extra_routes.append(
                {
                    "exchange": extra_route["exchange"],
                    "symbol": extra_route["symbol"],
                    "timeframe": extra_route["timeframe"],
                }
            )

    return candles, extra_routes


def get_candles_with_cache(
    exchange: str, symbol: str, start_date: str, finish_date: str
) -> np.ndarray:
    path = pathlib.Path("storage/bulk")
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f"storage/bulk/{cache_file_name}")

    if cache_file.is_file():
        with open(f"storage/bulk/{cache_file_name}", "rb") as handle:
            candles = pickle.load(handle)
            print(f"Loaded {len(candles) if candles is not None else 0} cached candles for {exchange} {symbol}")
            return candles

    # Try to find existing Jesse candle files first
    start_timestamp = jh.date_to_timestamp(start_date)
    finish_timestamp = jh.date_to_timestamp(finish_date)
    
    # Look for Jesse's candle pickle files in storage/temp
    temp_path = pathlib.Path("storage/temp")
    if temp_path.exists():
        for pickle_file in temp_path.glob("*.pickle"):
            if exchange in pickle_file.name and symbol in pickle_file.name:
                try:
                    # Extract timestamps from filename
                    parts = pickle_file.stem.split("-")
                    if len(parts) >= 2:
                        file_start_ts = int(parts[0])
                        file_end_ts = int(parts[1])
                        
                        # Check if this file covers our date range
                        if file_start_ts <= start_timestamp and file_end_ts >= finish_timestamp:
                            print(f"Found existing Jesse candle file: {pickle_file}")
                            with open(pickle_file, "rb") as f:
                                all_candles = pickle.load(f)
                            
                            # Filter candles to our date range
                            filtered_candles = []
                            for candle in all_candles:
                                if start_timestamp <= candle[0] <= finish_timestamp:
                                    filtered_candles.append(candle)
                            
                            candles = np.array(filtered_candles)
                            print(f"Filtered to {len(candles)} candles for date range {start_date} to {finish_date}")
                            
                            # Cache for future use
                            with open(f"storage/bulk/{cache_file_name}", "wb") as handle:
                                pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            return candles
                except Exception as e:
                    print(f"Error reading Jesse candle file {pickle_file}: {e}")
                    continue

    # Fall back to Jesse's get_candles function
    try:
        print(f"Fetching candles for {exchange} {symbol} from {start_date} to {finish_date}")
        candles = get_candles(exchange, symbol, "1m", start_timestamp, finish_timestamp)
        print(f"Fetched {len(candles) if candles is not None else 0} candles")
        with open(f"storage/bulk/{cache_file_name}", "wb") as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return candles
    except Exception as e:
        print(f"Error fetching candles for {exchange} {symbol}: {e}")
        return np.array([])


def _get_candles_with_warmup(
    exchange: str, symbol: str, start_date: str, finish_date: str, warmup_candles: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get candles with warmup period included.
    
    Returns:
        Tuple of (warmup_candles, trading_candles)
    """
    # Calculate warmup start date
    start_ts = jh.date_to_timestamp(start_date)
    warmup_start_ts = start_ts - (warmup_candles * 60_000)  # 1m candles
    warmup_start_date = jh.timestamp_to_date(warmup_start_ts)
    
    # Get all candles including warmup
    all_candles = get_candles_with_cache(exchange, symbol, warmup_start_date, finish_date)
    
    # Check if we got valid candles
    if all_candles is None or len(all_candles) == 0:
        print(f"Warning: No candles found for {exchange} {symbol} from {warmup_start_date} to {finish_date}")
        return np.array([]), np.array([])
    
    # Find the split point
    split_index = 0
    for i, candle in enumerate(all_candles):
        if candle is not None and candle[0] >= start_ts:
            split_index = i
            break
    
    # Split into warmup and trading candles
    warmup = all_candles[:split_index] if split_index > 0 else np.array([])
    trading = all_candles[split_index:] if split_index < len(all_candles) else all_candles
    
    return warmup, trading


def _decode_dna(hp_dict, dna):
    """
    Decode DNA string to hyperparameters (base64 format only).
    """
    if not dna:
        return None
        
    import base64
    import json
    decoded = base64.b64decode(dna.encode()).decode()
    return json.loads(decoded)


def backtest_with_info_key(
    key, config, route, extra_routes, start_date, end_date, hp_dict, dna
):
    hp = _decode_dna(hp_dict, dna)
    got_exception = False

    try:
        # Use Jesse's research.get_candles to get candles properly
        from jesse.research import get_candles as research_get_candles
        
        # Convert dates to timestamps
        start_timestamp = jh.date_to_timestamp(start_date)
        end_timestamp = jh.date_to_timestamp(end_date)
        warmup_num = config.get('warm_up_candles', 240)
        
        # Calculate correct warmup for the timeframe
        # warmup_num is in terms of the strategy's timeframe, but research.get_candles expects 1m candles
        main_timeframe = route[0]['timeframe']
        timeframe_minutes = jh.timeframe_to_one_minutes(main_timeframe)
        warmup_1m_candles = warmup_num * timeframe_minutes
        
        # Prepare candles and warmup_candles dictionaries
        candles = {}
        warmup_candles = {}
        
        # Load main route candles
        main_exchange = route[0]['exchange']
        main_symbol = route[0]['symbol']
        
        # Use Jesse's research.get_candles function
        warmup_arr, trading_arr = research_get_candles(
            main_exchange, 
            main_symbol, 
            '1m',  # Always use 1m candles
            start_timestamp, 
            end_timestamp,
            warmup_candles_num=warmup_1m_candles,  # Use calculated 1m candles
            caching=True
        )
        
        key_str = jh.key(main_exchange, main_symbol)
        candles[key_str] = {
            'exchange': main_exchange,
            'symbol': main_symbol,
            'candles': trading_arr
        }
        
        if warmup_arr is not None and len(warmup_arr) > 0:
            warmup_candles[key_str] = {
                'exchange': main_exchange,
                'symbol': main_symbol,
                'candles': warmup_arr
            }
        
        # Load extra route candles if any
        for extra_route in extra_routes:
            extra_exchange = extra_route['exchange']
            extra_symbol = extra_route['symbol']
            
            # Calculate warmup for extra route timeframe
            extra_timeframe = extra_route.get('timeframe', main_timeframe)
            extra_timeframe_minutes = jh.timeframe_to_one_minutes(extra_timeframe)
            extra_warmup_1m_candles = warmup_num * extra_timeframe_minutes
            
            extra_warmup, extra_trading = research_get_candles(
                extra_exchange,
                extra_symbol,
                '1m',
                start_timestamp,
                end_timestamp,
                warmup_candles_num=extra_warmup_1m_candles,
                caching=True
            )
            
            key_str = jh.key(extra_exchange, extra_symbol)
            candles[key_str] = {
                'exchange': extra_exchange,
                'symbol': extra_symbol,
                'candles': extra_trading
            }
            
            if extra_warmup is not None and len(extra_warmup) > 0:
                warmup_candles[key_str] = {
                    'exchange': extra_exchange,
                    'symbol': extra_symbol,
                    'candles': extra_warmup
                }
        
        # Prepare routes and data_routes in Jesse's expected format
        routes = []
        for r in route:
            routes.append({
                'exchange': r['exchange'],
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
                'strategy': r['strategy']
            })
        
        data_routes = []
        for r in extra_routes:
            data_routes.append({
                'exchange': r['exchange'],
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
            })
        
        # Call Jesse's research.backtest function
        result = backtest(
            config=config,
            routes=routes,
            data_routes=data_routes,
            candles=candles,
            warmup_candles=warmup_candles if warmup_candles else None,
            hyperparameters=hp,
            fast_mode=True
        )
        backtest_data = result["metrics"]
        
    except Exception as e:
        logger = start_logger_if_necessary()
        logger.error(
            "".join(traceback.TracebackException.from_exception(e).format()),
            extra={"key": key},
        )
        got_exception = True
        backtest_data = {}

    if got_exception or backtest_data.get("total", 0) == 0:
        backtest_data = {
            "total": 0,
            "total_winning_trades": None,
            "total_losing_trades": None,
            "starting_balance": None,
            "finishing_balance": None,
            "win_rate": None,
            "ratio_avg_win_loss": None,
            "longs_count": None,
            "longs_percentage": None,
            "shorts_percentage": None,
            "shorts_count": None,
            "fee": None,
            "net_profit": None,
            "net_profit_percentage": None,
            "average_win": None,
            "average_loss": None,
            "expectancy": None,
            "expectancy_percentage": None,
            "expected_net_profit_every_100_trades": None,
            "average_holding_period": None,
            "average_winning_holding_period": None,
            "average_losing_holding_period": None,
            "gross_profit": None,
            "gross_loss": None,
            "max_drawdown": None,
            "annual_return": None,
            "sharpe_ratio": None,
            "calmar_ratio": None,
            "sortino_ratio": None,
            "omega_ratio": None,
            "serenity_index": None,
            "smart_sharpe": None,
            "smart_sortino": None,
            "total_open_trades": None,
            "open_pl": None,
            "winning_streak": None,
            "losing_streak": None,
            "largest_losing_trade": None,
            "largest_winning_trade": None,
            "current_streak": None,
        }
        if got_exception:
            backtest_data["total"] = "error"

    return {**{"key": key}, **backtest_data}


def get_config():
    cfg_file = pathlib.Path("bulk_config.yml")

    if not cfg_file.is_file():
        print("bulk_config not found. Run create-config command.")
        exit()
    else:
        with open("bulk_config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

    return cfg
