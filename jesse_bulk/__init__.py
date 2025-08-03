import warnings
# Suppress pkg_resources deprecation warnings from Jesse (must be before Jesse imports)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import datetime
import hashlib
import logging
import os
import pathlib
import pickle
import random
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
@click.argument("strategy_name", required=False)
@click.option("--top-n", default=10, help="Number of top DNAs to test (default: 10)")
@click.option("--runs-per-dna", default=5, help="Number of runs per DNA (default: 5)")
@click.option("--timeframes", multiple=True, help="Additional timeframes to test (e.g., --timeframes 1h --timeframes 4h)")
@click.option("--export-csv", help="Export hall of fame to CSV file")
@click.option("--min-sharpe", type=float, help="Minimum Sharpe ratio filter")
@click.option("--min-profit", type=float, help="Minimum profit percentage filter")
@click.option("--show-stats", is_flag=True, help="Show hall of fame statistics")
def hall_of_fame(strategy_name: Optional[str], top_n: int, runs_per_dna: int, 
                timeframes: Tuple[str], export_csv: Optional[str],
                min_sharpe: Optional[float], min_profit: Optional[float],
                show_stats: bool) -> None:
    """Test the best DNAs from the Hall of Fame across multiple time periods and timeframes"""
    from .hall_of_fame import get_hall_of_fame
    
    validate_cwd()
    cfg = get_config()
    
    # Get hall of fame instance
    hof = get_hall_of_fame()
    
    # Show statistics if requested
    if show_stats:
        stats = hof.get_statistics()
        print("\n🏆 HALL OF FAME STATISTICS")
        print("="*50)
        print(f"Total DNAs: {stats['total_dnas']}")
        
        if stats['total_dnas'] > 0:
            print(f"\nStrategies:")
            for strategy, count in stats['by_strategy'].items():
                print(f"  - {strategy}: {count} DNAs")
            
            avg_metrics = stats['average_metrics']
            print(f"\nAverage Metrics:")
            if avg_metrics['sharpe_ratio'] is not None:
                print(f"  - Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
            if avg_metrics['net_profit_percentage'] is not None:
                print(f"  - Net Profit: {avg_metrics['net_profit_percentage']:.1f}%")
            if avg_metrics['win_rate'] is not None:
                print(f"  - Win Rate: {avg_metrics['win_rate']:.1%}")
            if avg_metrics['max_drawdown'] is not None:
                print(f"  - Max Drawdown: {avg_metrics['max_drawdown']:.1f}%")
            
            print(f"\nTop Performers:")
            if stats['top_sharpe'] is not None:
                print(f"  - Best Sharpe: {stats['top_sharpe']:.2f}")
            if stats['top_profit'] is not None:
                print(f"  - Best Profit: {stats['top_profit']:.1f}%")
            print(f"  - Total Tests: {stats['total_tests']}")
        else:
            print("\nNo DNAs in Hall of Fame yet. Run 'refine-best' to add top performers.")
        
        print("="*50)
        
        if not strategy_name and not export_csv:
            return
    
    # Export if requested
    if export_csv:
        hof.export_to_csv(export_csv, strategy_name)
        return
    
    # Get best DNAs from hall of fame
    best_dnas_df = hof.get_best_dnas(
        strategy_name=strategy_name,
        limit=top_n,
        min_trades=20  # Ensure meaningful results
    )
    
    if best_dnas_df.empty:
        print("No DNAs found in Hall of Fame matching criteria")
        return
    
    # Apply additional filters
    if min_sharpe:
        best_dnas_df = best_dnas_df[best_dnas_df['sharpe_ratio'] >= min_sharpe]
    if min_profit:
        best_dnas_df = best_dnas_df[best_dnas_df['net_profit_percentage'] >= min_profit]
    
    if best_dnas_df.empty:
        print("No DNAs found after applying filters")
        return
    
    print(f"\n🏆 TESTING TOP {len(best_dnas_df)} HALL OF FAME DNAs")
    print("="*60)
    
    # Show selected DNAs
    print("\nSelected DNAs:")
    for idx, row in best_dnas_df.iterrows():
        print(f"\n#{idx+1} - {row['strategy_name']} ({row['symbol']} {row['timeframe']})")
        print(f"  - Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"  - Profit: {row['net_profit_percentage']:.1f}%") 
        print(f"  - Win Rate: {row['win_rate']:.1%}")
        print(f"  - Discovered: {row['discovery_date'][:10]}")
    
    # Use the first DNA's strategy if not specified
    if not strategy_name:
        strategy_name = best_dnas_df.iloc[0]['strategy_name']
        print(f"\nUsing strategy: {strategy_name}")
    
    # Prepare backtests
    StrategyClass = jh.get_strategy_class(strategy_name)
    hp_dict = StrategyClass().hyperparameters()
    
    # Prepare config
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
    
    # Determine timeframes to test
    base_timeframes = list(set(best_dnas_df['timeframe'].unique()))
    test_timeframes = list(set(base_timeframes + list(timeframes)))
    
    print(f"\nTimeframes to test: {', '.join(test_timeframes)}")
    print(f"Runs per DNA per timeframe: {runs_per_dna}")
    
    # Generate test combinations
    mp_args = []
    for _, dna_row in best_dnas_df.iterrows():
        dna_str = dna_row['dna']
        
        for timeframe in test_timeframes:
            for run in range(runs_per_dna):
                # Generate random dates
                warm_up_days = cfg["backtest-data"]["warm_up_candles"]
                end_date = cfg["backtest-data"].get("end_date", datetime.now().strftime("%Y-%m-%d"))
                start_date = get_random_dates_within_timespan(end_date, warm_up_days)
                
                # Use the DNA's original symbol
                symbol = dna_row['symbol']
                
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
                
                key = f'{symbol}-{timeframe}-{start_date}-{end_date}-{dna_str}'
                mp_args.append((
                    key,
                    config,
                    route,
                    extra_routes,
                    start_date,
                    end_date,
                    hp_dict,
                    dna_str,
                ))
    
    # Run all backtests in parallel
    n_jobs = joblib.cpu_count() if cfg["n_jobs"] == -1 else cfg["n_jobs"]
    print(f"\nStarting {len(mp_args)} backtests...")
    parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
    all_results = parallel(
        joblib.delayed(backtest_with_info_key)(*args) for args in mp_args
    )
    
    # Process results
    results_df = pd.DataFrame.from_dict(all_results, orient="columns")
    results_df["dna"] = results_df["key"].apply(lambda x: x.split("-")[-1])
    results_df["timeframe"] = results_df["key"].apply(lambda x: x.split("-")[2])
    
    # Calculate statistics by DNA and timeframe
    print("\n" + "="*60)
    print("HALL OF FAME TEST RESULTS")
    print("="*60)
    
    # Group by DNA
    for dna_str in best_dnas_df['dna']:
        dna_results = results_df[results_df['dna'] == dna_str]
        if dna_results.empty:
            continue
            
        original_row = best_dnas_df[best_dnas_df['dna'] == dna_str].iloc[0]
        dna_hash = hashlib.sha256(dna_str.encode()).hexdigest()[:16]
        
        print(f"\n📊 DNA: ...{dna_hash[-8:]}")
        print(f"   Strategy: {original_row['strategy_name']}")
        print(f"   Original: {original_row['sharpe_ratio']:.2f} Sharpe, {original_row['net_profit_percentage']:.1f}% profit")
        
        # Results by timeframe
        for tf in sorted(dna_results['timeframe'].unique()):
            tf_results = dna_results[dna_results['timeframe'] == tf]
            
            avg_sharpe = tf_results['sharpe_ratio'].mean() if 'sharpe_ratio' in tf_results and not tf_results.empty else 0
            avg_profit = tf_results['net_profit_percentage'].mean() if 'net_profit_percentage' in tf_results and not tf_results.empty else 0
            avg_winrate = tf_results['win_rate'].mean() if 'win_rate' in tf_results and not tf_results.empty else 0
            
            # Calculate success rate based on presence of key metrics
            # Successful runs have 'sharpe_ratio', failed runs might have 'status' or missing values
            if 'sharpe_ratio' in tf_results.columns:
                # Count rows where sharpe_ratio is not null (successful runs)
                successful_runs = tf_results['sharpe_ratio'].notna().sum()
                total_runs = len(tf_results)
                success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            else:
                success_rate = 0  # All failed if no sharpe_ratio column
            
            print(f"\n   {tf} timeframe ({len(tf_results)} runs):")
            print(f"     - Success Rate: {success_rate:.0f}%")
            print(f"     - Avg Sharpe: {avg_sharpe:.2f}")
            print(f"     - Avg Profit: {avg_profit:.1f}%")
            print(f"     - Avg Win Rate: {avg_winrate:.1%}")
            
            # Store performance history
            for _, result in tf_results.iterrows():
                if 'status' in result and result.get('status') == 'success':
                    test_conditions = {
                        'symbol': result['key'].split('-')[0] + '-' + result['key'].split('-')[1],
                        'timeframe': tf,
                        'start_date': result['key'].split('-')[3],
                        'end_date': result['key'].split('-')[4],
                    }
                    
                    hof.add_performance_test(
                        dna_hash=dna_hash,
                        test_results=result.to_dict(),
                        test_conditions=test_conditions,
                        test_type='hall_of_fame_test'
                    )
    
    # Save detailed results
    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    results_df.to_csv(
        f"hall_of_fame_test_{dt}.csv", encoding="utf-8", sep="\t"
    )
    
    print(f"\n💾 Detailed results saved to: hall_of_fame_test_{dt}.csv")
    print(f"📊 Performance history updated in Hall of Fame database")


@cli.command()
def presets() -> None:
    """Show available selection presets and their descriptions"""
    from .selection import SELECTION_PRESETS
    
    print("Available Selection Presets:")
    print("="*50)
    
    preset_descriptions = {
        'conservative': {
            'description': 'Focuses on risk-adjusted returns and consistent performance',
            'targets': 'High Sharpe ratio, good win rate, low drawdown',
            'best_for': 'Risk-averse traders who prioritize capital preservation'
        },
        'aggressive': {
            'description': 'Targets maximum profit potential and high expectancy',
            'targets': 'High net profit, strong expectancy, good win/loss ratio',
            'best_for': 'Profit-focused traders willing to accept higher risk'
        },
        'balanced': {
            'description': 'Balances profit, risk, and consistency with diversity',
            'targets': 'Multiple metrics with parameter diversity',
            'best_for': 'Most traders seeking well-rounded strategies'
        },
        'robust': {
            'description': 'Uses Pareto optimization for multi-objective selection',
            'targets': 'Training vs testing performance balance',
            'best_for': 'Advanced users wanting non-dominated solutions'
        }
    }
    
    for preset_name, config in SELECTION_PRESETS.items():
        desc = preset_descriptions.get(preset_name, {})
        print(f"\n🎯 {preset_name.upper()}")
        print(f"   Description: {desc.get('description', 'No description available')}")
        print(f"   Targets: {desc.get('targets', 'N/A')}")
        print(f"   Best for: {desc.get('best_for', 'N/A')}")
        
        print(f"   Metrics ({len(config['metrics'])}):")
        for metric in config['metrics']:
            weight_pct = metric['weight'] * 100
            thresholds = []
            if metric.get('min_threshold'):
                thresholds.append(f"min≥{metric['min_threshold']}")
            if metric.get('max_threshold'):
                thresholds.append(f"max≤{metric['max_threshold']}")
            threshold_str = f" ({', '.join(thresholds)})" if thresholds else ""
            
            print(f"     - {metric['metric']}: {weight_pct:.0f}%{threshold_str}")
    
    print(f"\n💡 Usage Examples:")
    print(f"   jesse-bulk refine-best study.db --selection-preset conservative")
    print(f"   jesse-bulk refine-best study.db --selection-preset aggressive --top-n 10")


@cli.command()
@click.argument("db_path", required=True, type=str)
@click.option("--top-n", default=10, help="Number of top DNAs to test (default: 10)")
@click.option("--runs-per-dna", default=10, help="Number of runs per DNA (default: 10)")
@click.option("--selection-preset", type=click.Choice(['conservative', 'aggressive', 'balanced', 'robust']), 
              help="Use a preset selection strategy")
@click.option("--all-presets", is_flag=True, help="Test all selection presets and find the ultimate king DNA")
def refine_best(db_path: str, top_n: int, runs_per_dna: int, selection_preset: Optional[str] = None, all_presets: bool = False) -> None:
    from .optuna_reader import read_optuna_study, find_optuna_databases
    from .selection import SelectionEngine, SELECTION_PRESETS
    
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
    
    # Use advanced selection if configured, otherwise fall back to simple selection
    selection_config = cfg.get('selection_strategy', None)
    
    # Store both selection results for comparison
    advanced_dnas = None
    simple_dnas = None
    
    # Handle --all-presets flag
    if all_presets:
        print("\n" + "="*60)
        print("🏆 TESTING ALL SELECTION PRESETS")
        print("="*60)
        print(f"\nWill test {len(SELECTION_PRESETS)} presets to find the ultimate king DNA")
        
        all_preset_results = {}
        all_preset_dnas = {}
        
        # Run each preset
        for preset_name in SELECTION_PRESETS:
            print(f"\n{'='*60}")
            print(f"Testing preset: {preset_name.upper()}")
            print(f"{'='*60}")
            
            selection_config = SELECTION_PRESETS[preset_name]
            engine = SelectionEngine(selection_config)
            preset_dnas = engine.select(dna_df, top_n)
            
            # Store selected DNAs
            all_preset_dnas[preset_name] = preset_dnas
            
            # Print selection stats
            report = engine.get_selection_report(dna_df, preset_dnas)
            print(f"\nSelected {len(preset_dnas)} DNAs")
            if 'metric_stats' in report:
                for metric, stats in report['metric_stats'].items():
                    print(f"  - {metric}: {stats['min']:.2f} to {stats['max']:.2f} (avg: {stats['mean']:.2f})")
        
        # Combine all unique DNAs from all presets and track their origin
        for preset_name, preset_dnas in all_preset_dnas.items():
            preset_dnas['preset_origin'] = preset_name
        
        all_unique_dnas = pd.concat(all_preset_dnas.values())
        # Keep track of which presets selected each DNA
        dna_preset_map = all_unique_dnas.groupby('dna')['preset_origin'].apply(list).to_dict()
        all_unique_dnas = all_unique_dnas.drop_duplicates(subset=['dna'])
        
        print(f"\n{'='*60}")
        print(f"Total unique DNAs across all presets: {len(all_unique_dnas)}")
        
        # Show overlap between presets
        dna_counts = {}
        for dna, presets in dna_preset_map.items():
            count = len(presets)
            if count not in dna_counts:
                dna_counts[count] = 0
            dna_counts[count] += 1
        
        print("\nDNA selection overlap:")
        for count, num_dnas in sorted(dna_counts.items(), reverse=True):
            if count > 1:
                print(f"  - {num_dnas} DNAs selected by {count} presets")
            else:
                print(f"  - {num_dnas} DNAs selected by only 1 preset")
        print(f"{'='*60}")
        
        # Use combined DNAs for testing
        top_dnas = all_unique_dnas
        
    # Override with preset if specified
    elif selection_preset:
        selection_config = SELECTION_PRESETS[selection_preset]
        print(f"Using '{selection_preset}' selection preset")
        
        # Always run both selections for comparison when using preset
        print("\n" + "="*60)
        print("ADVANCED MULTI-CRITERIA SELECTION")
        print("="*60)
        engine = SelectionEngine(selection_config)
        advanced_dnas = engine.select(dna_df, top_n)
        
        # Print selection report
        report = engine.get_selection_report(dna_df, advanced_dnas)
        print(f"\nSelection Report:")
        print(f"  - Total candidates: {report['total_candidates']}")
        print(f"  - Selected: {report['total_selected']}")
        print(f"  - Metrics used: {', '.join(report['metrics_used'])}")
        
        if 'metric_stats' in report:
            print("\nSelected DNA statistics:")
            for metric, stats in report['metric_stats'].items():
                print(f"  - {metric}: {stats['min']:.2f} to {stats['max']:.2f} (avg: {stats['mean']:.2f})")
        
        # Also run simple selection for comparison
        print("\n" + "="*60)
        print("SIMPLE SELECTION (for comparison)")
        print("="*60)
        sort_criteria = cfg.get('sort_by', 'training_log.net_profit_percentage')
        
        if sort_criteria in dna_df.columns:
            simple_dnas = dna_df.nlargest(top_n, sort_criteria)
            print(f"Sorted by: {sort_criteria}")
            
            # Show simple selection statistics
            if sort_criteria in simple_dnas.columns:
                print(f"\nSimple selection {sort_criteria} stats:")
                print(f"  - Range: {simple_dnas[sort_criteria].min():.2f} to {simple_dnas[sort_criteria].max():.2f}")
                print(f"  - Average: {simple_dnas[sort_criteria].mean():.2f}")
        
        # Compare the selections
        print("\n" + "="*60)
        print("HEAD-TO-HEAD COMPARISON")
        print("="*60)
        
        # Find overlapping DNAs
        advanced_set = set(advanced_dnas['dna'])
        simple_set = set(simple_dnas['dna'])
        overlap = advanced_set.intersection(simple_set)
        
        print(f"\nDNA Selection Overlap:")
        print(f"  - Advanced selection: {len(advanced_dnas)} unique DNAs")
        print(f"  - Simple selection: {len(simple_dnas)} unique DNAs")
        print(f"  - Overlap: {len(overlap)} DNAs ({len(overlap)/len(advanced_dnas)*100:.1f}% of advanced)")
        
        # Compare key metrics
        comparison_metrics = ['training_log.sharpe_ratio', 'training_log.net_profit_percentage', 
                            'training_log.win_rate', 'training_log.max_drawdown']
        
        print("\nAverage Metrics Comparison:")
        print(f"{'Metric':<35} {'Advanced':>12} {'Simple':>12} {'Difference':>12}")
        print("-" * 72)
        
        for metric in comparison_metrics:
            if metric in advanced_dnas.columns and metric in simple_dnas.columns:
                adv_avg = advanced_dnas[metric].mean()
                simple_avg = simple_dnas[metric].mean()
                diff = adv_avg - simple_avg
                diff_pct = (diff / abs(simple_avg) * 100) if simple_avg != 0 else 0
                
                # Format based on metric type
                if 'percentage' in metric or 'rate' in metric:
                    print(f"{metric:<35} {adv_avg:>11.2f}% {simple_avg:>11.2f}% {diff_pct:>11.1f}%")
                else:
                    print(f"{metric:<35} {adv_avg:>12.2f} {simple_avg:>12.2f} {diff_pct:>11.1f}%")
        
        # Use advanced selection for actual testing
        top_dnas = advanced_dnas
        print(f"\n{'='*60}")
        print(f"Using ADVANCED selection results for backtesting")
        print(f"{'='*60}")
        
    elif selection_config:
        # Use advanced selection engine without comparison
        print("Using advanced multi-criteria selection")
        engine = SelectionEngine(selection_config)
        top_dnas = engine.select(dna_df, top_n)
        
        # Print selection report
        report = engine.get_selection_report(dna_df, top_dnas)
        print(f"\nSelection Report:")
        print(f"  - Total candidates: {report['total_candidates']}")
        print(f"  - Selected: {report['total_selected']}")
        print(f"  - Metrics used: {', '.join(report['metrics_used'])}")
        
        if 'metric_stats' in report:
            print("\nSelected DNA statistics:")
            for metric, stats in report['metric_stats'].items():
                print(f"  - {metric}: {stats['min']:.2f} to {stats['max']:.2f} (avg: {stats['mean']:.2f})")
    else:
        # Fall back to simple selection
        sort_criteria = cfg.get('sort_by', 'training_log.net_profit_percentage')
        
        if sort_criteria in dna_df.columns:
            top_dnas = dna_df.nlargest(top_n, sort_criteria)
            print(f"Simple selection sorted by: {sort_criteria}")
        else:
            print(f"Warning: Sort column '{sort_criteria}' not found in database.")
            print(f"Available columns include: {[col for col in dna_df.columns if 'net_profit_percentage' in col]}")
            top_dnas = dna_df.head(top_n)
    
    print(f"\nSelected {len(top_dnas)} DNAs for refinement")

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
    
    # If using --all-presets, show which preset found the king DNA
    if all_presets:
        print("\n" + "="*60)
        print("🏆 PRESET PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Map DNAs back to their presets
        preset_performance = {}
        for dna_str in sorted_results.index:
            if dna_str in dna_preset_map:
                for preset in dna_preset_map[dna_str]:
                    if preset not in preset_performance:
                        preset_performance[preset] = []
                    preset_performance[preset].append({
                        'dna': dna_str,
                        'finishing_balance': sorted_results.loc[dna_str, 'finishing_balance'],
                        'sharpe_ratio': sorted_results.loc[dna_str, 'sharpe_ratio'] if 'sharpe_ratio' in sorted_results else 0,
                        'win_rate': sorted_results.loc[dna_str, 'win_rate'] if 'win_rate' in sorted_results else 0
                    })
        
        # Analyze each preset's performance
        preset_stats = {}
        for preset, dnas in preset_performance.items():
            if dnas:
                avg_balance = np.mean([d['finishing_balance'] for d in dnas])
                max_balance = max([d['finishing_balance'] for d in dnas])
                avg_sharpe = np.mean([d['sharpe_ratio'] for d in dnas])
                preset_stats[preset] = {
                    'avg_finishing_balance': avg_balance,
                    'max_finishing_balance': max_balance,
                    'avg_sharpe': avg_sharpe,
                    'dna_count': len(dnas),
                    'best_dna': max(dnas, key=lambda x: x['finishing_balance'])
                }
        
        # Sort presets by average performance
        sorted_presets = sorted(preset_stats.items(), key=lambda x: x[1]['avg_finishing_balance'], reverse=True)
        
        print("\nPreset Rankings (by average finishing balance):")
        for i, (preset, stats) in enumerate(sorted_presets, 1):
            print(f"\n{i}. {preset.upper()}")
            print(f"   - DNAs tested: {stats['dna_count']}")
            print(f"   - Avg finishing balance: {stats['avg_finishing_balance']:.2f}%")
            print(f"   - Best finishing balance: {stats['max_finishing_balance']:.2f}%")
            print(f"   - Avg Sharpe ratio: {stats['avg_sharpe']:.2f}")
        
        # Find which preset(s) selected the overall king DNA
        king_dna = best_dna.name
        king_presets = dna_preset_map.get(king_dna, [])
        
        print("\n" + "-"*60)
        print(f"👑 THE KING DNA was selected by: {', '.join([p.upper() for p in king_presets])}")
        print(f"   DNA: {king_dna[:20]}...")
        print(f"   Finishing Balance: {best_dna['finishing_balance']:.2f}%")
        if 'sharpe_ratio' in best_dna:
            print(f"   Sharpe Ratio: {best_dna['sharpe_ratio']:.2f}")
        if 'win_rate' in best_dna:
            print(f"   Win Rate: {best_dna['win_rate']:.2%}")
        print("-"*60)
    
    # Add top performers to Hall of Fame
    from .hall_of_fame import get_hall_of_fame
    hall_of_fame = get_hall_of_fame()
    
    print("\n" + "="*60)
    print("ADDING TOP PERFORMERS TO HALL OF FAME")
    print("="*60)
    
    # Get detailed results for each DNA
    for dna_str in sorted_results.head(3).index:  # Top 3 DNAs
        # Get all results for this DNA
        dna_results = results_df[results_df['dna'] == dna_str]
        
        # Calculate aggregate metrics
        def safe_mean(series, convert_int=False):
            try:
                # Filter out non-numeric values
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().any():
                    mean_val = numeric_series.mean()
                    return int(mean_val) if convert_int and pd.notna(mean_val) else mean_val
            except:
                pass
            return None
            
        metrics = {
            'sharpe_ratio': safe_mean(dna_results.get('sharpe_ratio')),
            'net_profit_percentage': safe_mean(dna_results.get('net_profit_percentage')),
            'win_rate': safe_mean(dna_results.get('win_rate')),
            'max_drawdown': safe_mean(dna_results.get('max_drawdown')),
            'total': safe_mean(dna_results.get('total'), convert_int=True),
            'expectancy_percentage': safe_mean(dna_results.get('expectancy_percentage')),
            'ratio_avg_win_loss': safe_mean(dna_results.get('ratio_avg_win_loss')),
            'calmar_ratio': safe_mean(dna_results.get('calmar_ratio')),
            'sortino_ratio': safe_mean(dna_results.get('sortino_ratio')),
            'omega_ratio': safe_mean(dna_results.get('omega_ratio')),
            'finishing_balance': safe_mean(dna_results.get('finishing_balance')),
        }
        
        # Get test conditions from the first result
        first_result_key = dna_results.iloc[0]['key']
        key_parts = first_result_key.split('-')
        
        test_conditions = {
            'symbol': key_parts[0] + '-' + key_parts[1],  # e.g., BTC-USDT
            'timeframe': key_parts[2],
            'start_date': key_parts[3],
            'end_date': key_parts[4],
        }
        
        try:
            record_id = hall_of_fame.add_dna(
                dna=dna_str,
                metrics=metrics,
                strategy_name=strategy_name,
                test_conditions=test_conditions,
                source='refine-best',
                selection_method=selection_preset if selection_preset else 'custom',
                notes=f"Avg of {len(dna_results)} runs"
            )
            
            print(f"✅ Added DNA to Hall of Fame (ID: {record_id})")
            print(f"   - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   - Profit: {metrics.get('net_profit_percentage', 0):.2f}%")
            print(f"   - Win Rate: {metrics.get('win_rate', 0):.2%}")
            
        except Exception as e:
            print(f"❌ Failed to add DNA to Hall of Fame: {e}")
    
    # Show Hall of Fame statistics
    stats = hall_of_fame.get_statistics()
    print(f"\n📊 Hall of Fame Statistics:")
    print(f"   - Total DNAs: {stats['total_dnas']}")
    if stats['total_dnas'] > 0:
        print(f"   - Strategies: {', '.join(f'{k}({v})' for k, v in stats['by_strategy'].items())}")
        if stats['average_metrics']['sharpe_ratio'] is not None:
            print(f"   - Avg Sharpe: {stats['average_metrics']['sharpe_ratio']:.2f}")
        if stats['top_sharpe'] is not None:
            print(f"   - Top Sharpe: {stats['top_sharpe']:.2f}")
        if stats['top_profit'] is not None:
            print(f"   - Top Profit: {stats['top_profit']:.2f}%")


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
