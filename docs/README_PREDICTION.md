# ZIP Code Home Price Prediction System

## Installation and Setup

### Data Acquisition
Download the latest Zillow Home Value Index data:
```bash
python scripts/download_data.py
```

### Forecast Generation
Execute a 6-month forecast for a specified ZIP code:
```bash
python scripts/predict_zip.py --zip 90210 --months 6
```

## Implementation Examples

### Standard Forecast
```bash
# 6-month forecast for Beverly Hills, CA
python scripts/predict_zip.py --zip 90210 --months 6
```

### Multiple Forecast Horizons
```bash
# 3, 6, and 12-month forecasts for Katy, TX
python scripts/predict_zip.py --zip 77449 --months 3,6,12
```

### Custom Confidence Intervals
```bash
# 90% confidence intervals
python scripts/predict_zip.py --zip 11368 --months 12 --confidence 0.9
```

### Output Serialization
```bash
# Serialize results to JSON format
python scripts/predict_zip.py --zip 10001 --months 6,12 --output manhattan_forecast.json
```

## Example Forecast Results

**Beverly Hills, CA (90210)** - 6 Month Horizon:
- Current Value: $5,219,227.69
- Predicted Value: $5,169,876.50
- Expected Change: -$49,351.19 (-0.9%)
- Market Classification: Stable Market Expected

**Katy, TX (77449)** - 12 Month Horizon:
- Current Value: $282,289.31
- Predicted Value: $272,786.88
- Expected Change: -$9,502.43 (-3.4%)
- Market Classification: Moderate Decline Expected

**Manhattan, NY (10001)** - 6 Month Horizon:
- Current Value: $1,693,336.07
- Predicted Value: $1,683,454.50
- Expected Change: -$9,881.57 (-0.6%)
- Market Classification: Stable Market Expected

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--zip` | string | required | 5-digit ZIP code identifier |
| `--months` | int/list | 6 | Forecast horizon(s) in months |
| `--confidence` | float | 0.8 | Prediction interval confidence level |
| `--data-path` | string | `data/raw/zhvi_zip.csv` | ZHVI data file path |
| `--model-cache` | string | `data/model_cache` | Model cache directory |
| `--output` | string | None | JSON output file path |
| `--verbose` | flag | False | Extended logging output |

## Output Format

```
================================================================================
HOME PRICE FORECAST FOR ZIP CODE 90210
================================================================================

Location: Beverly Hills, CA
Metro Area: Los Angeles-Long Beach-Anaheim, CA
County: Los Angeles County

Data Coverage:
  Historical Data Points: 298 months
  Data Period: 2000-01-31 to 2024-10-31

FORECAST RESULTS:
--------------------------------------------------------------------------------

6-Month Forecast:
  Current Value:     $2,847,234.00
  Predicted Value:   $2,923,456.00
  Expected Change:   $76,222.00 (+2.7%)
  Confidence Range:  $2,856,789.00 - $2,990,123.00
  Confidence Level:  80%
  Market Outlook:    Moderate Appreciation Expected

================================================================================
IMPORTANT DISCLAIMERS:
• These predictions are based on historical data and statistical models
• Actual home prices may vary significantly due to local market conditions
• Economic factors, policy changes, and market events can affect outcomes
• This forecast should not be used as the sole basis for financial decisions
================================================================================
```

## Metric Definitions

### Forecast Components

- **Current Value**: Most recent ZHVI observation
- **Predicted Value**: Model point estimate for specified horizon
- **Expected Change**: Absolute and percentage price differential
- **Confidence Range**: Prediction interval bounds at specified confidence level
- **Confidence Level**: Statistical confidence in prediction interval

### Market Classification System

| Classification | Change Range | Interpretation |
|----------------|--------------|----------------|
| Strong Appreciation Expected | >+5% | Rapid price growth anticipated |
| Moderate Appreciation Expected | +2% to +5% | Steady price increases |
| Stable Market Expected | -2% to +2% | Minimal price movement |
| Moderate Decline Expected | -5% to -2% | Price softening anticipated |
| Significant Decline Expected | <-5% | Notable price decreases |

## Data Specifications

### Geographic Coverage
- **Total ZIP codes**: 26,316 across the United States
- **Modeling candidates**: ~15,000 with sufficient historical data
- **Data source**: Zillow Home Value Index (ZHVI)
- **Update frequency**: Monthly

### Data Quality Requirements
- **Historical depth**: Minimum 12 months of price history
- **Data recency**: Observations within the last 24 months
- **Data integrity**: Valid price ranges without critical missing periods

## Error Handling and Diagnostics

### Common Error Conditions

**ZIP code not found**
- Verify ZIP code format (5 digits)
- Rural or newly established ZIP codes may lack sufficient data
- Consider alternative ZIP codes in the same geographic area

**Insufficient historical data**
- ZIP code contains fewer than 12 months of observations
- Recommend broader geographic aggregation (county or metro level)

**Model initialization failure**
- Verify network connectivity for model download (~185MB)
- Ensure adequate disk space for model cache
- Clear model cache if corrupted: `rm -rf data/model_cache`

**Data file not found**
- Execute `python scripts/download_data.py` to acquire latest data
- Verify existence of `data/raw/zhvi_zip.csv`

### Diagnostic Procedures

1. **Enable verbose logging**: Add `--verbose` flag for detailed execution information
2. **Validate data coverage**: Execute `python src/data_processor.py` to enumerate available ZIP codes
3. **Verify ZIP code eligibility**: The system will report data sufficiency for specified ZIP codes

## Technical Architecture

### Model Specifications
- **Architecture**: Amazon Chronos T5 Small transformer
- **Training methodology**: Pre-training on diverse time series datasets
- **Context window**: Up to 512 historical observations
- **Forecast horizon**: 1-60 months (optimal: 1-24 months)

### Data Processing Pipeline
- **Source**: Zillow Home Value Index (ZHVI)
- **Frequency**: Monthly observations
- **Preprocessing**: Seasonally adjusted by Zillow
- **Scope**: Single-family homes, condominiums, cooperatives
- **Valuation methodology**: Automated Valuation Model (AVM)

### Statistical Methodology
- **Sampling approach**: 100 forecast trajectories per prediction
- **Confidence intervals**: Percentile-based from forecast distribution
- **Uncertainty quantification**: Standard deviation and prediction intervals
- **Validation**: Historical backtesting on out-of-sample data

## Model Limitations and Constraints

### Methodological Constraints
- **Historical dependency**: Predictions extrapolate from historical patterns
- **Exogenous factor insensitivity**: Cannot anticipate policy changes or economic shocks
- **Local event exclusion**: Neighborhood-specific developments not captured
- **Market regime sensitivity**: Performance varies across different market conditions

### Data Constraints
- **Geographic coverage**: Limited to areas with sufficient transaction volume
- **Property aggregation**: Combines different home types and sizes
- **Temporal lag**: Data may reflect 1-2 month reporting delays
- **Seasonal modeling**: Some seasonal patterns may not be fully captured

## Technical Dependencies

### Core Requirements
- Python 3.11+
- PyTorch 2.7.0
- Transformers 4.52.3
- Chronos-forecasting 1.5.2
- TensorFlow 2.19.0
- Pandas 2.2.3
- NumPy 1.26.4

---