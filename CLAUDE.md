# Jesse Bulk Project Context

This is a bulk backtesting tool for the Jesse trading framework. It provides utilities for running multiple backtests in parallel with different configurations.

## Key Components

- **jesse-bulk pick**: Processes optimization CSVs from Jesse, removes duplicates and filters based on config
- **jesse-bulk refine**: Runs backtests with specific DNAs from optimization CSV
- **jesse-bulk bulk**: Runs all backtests according to configuration

## Important Notes

- Only works with the dashboard version/branch of Jesse
- Uses joblib for multiprocessing
- Uses pickle cache for candles (stored in `storage/bulk`)
- Configuration is managed through `bulk_config.yml`
- Warm-up candles are taken from passed candles, so actual start_date differs from normal backtests
- Extra route candles are added to all backtests even if not needed

## File Structure

- `jesse_bulk/` - Main package directory
  - `picker.py` - Core logic for CSV processing and backtesting
  - `bulk_config.yml` - Configuration template
- `setup.py` - Package setup configuration

## Development Guidelines

- This is an older project that may need updates to work with newer Jesse versions
- Be mindful of the multiprocessing implementation when making changes
- Test thoroughly with different Jesse versions before committing changes