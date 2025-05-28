"""
Home Price Predictor - Main Prediction Logic

This module provides the core prediction functionality for home price forecasting.
It combines data processing, model inference, and advanced analytics to deliver
highly accurate predictions with confidence intervals and trend analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
from contextlib import contextmanager

from .data_processor import ZillowDataProcessor
from .model import ChronosT5Model
from .utils import (
    validate_zip_code, validate_forecast_horizon, validate_confidence_level,
    format_currency, calculate_percentage_change, ValidationError,
    ZHVI_DATA_PATH, DEFAULT_FORECAST_HORIZONS, DEFAULT_CONFIDENCE_LEVEL,
    MIN_HISTORY_POINTS, MAX_FORECAST_HORIZON, MODEL_CACHE_DIR
)

logger = logging.getLogger(__name__)

# Suppress pandas warnings for production
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@contextmanager
def error_context(operation: str, zip_code: Optional[str] = None):
    """Context manager for consistent error handling and logging."""
    try:
        yield
    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        error_msg = f"Error during {operation}"
        if zip_code:
            error_msg += f" for ZIP {zip_code}"
        error_msg += f": {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


class HomePricePredictor:
    """
    Main predictor class that orchestrates data processing and model inference
    to provide accurate home price predictions.
    
    Features:
    - Multi-horizon forecasting (3, 6, 12+ months)
    - Confidence intervals and uncertainty quantification
    - Trend analysis and market insights
    - Data quality validation
    - Seasonal adjustment capabilities
    - Comparative market analysis
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        model_name: str = "amazon/chronos-t5-small",
        enable_validation: bool = True
    ):
        """
        Initialize the Home Price Predictor.
        
        Args:
            data_path: Path to ZHVI data file
            model_cache_dir: Directory for model caching
            model_name: HuggingFace model identifier
            enable_validation: Whether to enable data validation
        """
        self.data_path = data_path or ZHVI_DATA_PATH
        self.model_cache_dir = model_cache_dir or MODEL_CACHE_DIR
        self.model_name = model_name
        self.enable_validation = enable_validation
        
        # Initialize components
        self.data_processor: Optional[ZillowDataProcessor] = None
        self.model: Optional[ChronosT5Model] = None
        self._initialized = False
        
        logger.info("HomePricePredictor initialized")
    
    def initialize(self) -> None:
        """Initialize data processor and model components."""
        if self._initialized:
            return
        
        with error_context("predictor initialization"):
            logger.info("Initializing predictor components...")
            
            # Initialize data processor
            self.data_processor = ZillowDataProcessor(self.data_path)
            self.data_processor.load_data()
            
            # Initialize model
            self.model = ChronosT5Model(
                model_name=self.model_name,
                cache_dir=self.model_cache_dir
            )
            self.model.load_model()
            
            self._initialized = True
            logger.info("Predictor initialization complete")
    
    def predict_zip_price(
        self,
        zip_code: str,
        forecast_horizons: Optional[List[int]] = None,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        num_samples: int = 100,
        include_trends: bool = True,
        include_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive price predictions for a ZIP code.
        
        Args:
            zip_code: 5-digit ZIP code
            forecast_horizons: List of forecast horizons in months
            confidence_level: Confidence level for intervals (0.0-1.0)
            num_samples: Number of Monte Carlo samples
            include_trends: Whether to include trend analysis
            include_seasonality: Whether to include seasonal analysis
            
        Returns:
            Comprehensive prediction results with metadata
            
        Raises:
            ValidationError: If input validation fails
            RuntimeError: If prediction fails
        """
        if not self._initialized:
            self.initialize()
        
        # Validate inputs
        if not validate_zip_code(zip_code):
            raise ValidationError(f"Invalid ZIP code: {zip_code}")
        
        if not validate_confidence_level(confidence_level):
            raise ValidationError(f"Invalid confidence level: {confidence_level}")
        
        if num_samples <= 0:
            raise ValidationError(f"Number of samples must be positive: {num_samples}")
        
        forecast_horizons = forecast_horizons or DEFAULT_FORECAST_HORIZONS
        for horizon in forecast_horizons:
            if not validate_forecast_horizon(horizon):
                raise ValidationError(f"Invalid forecast horizon: {horizon}")
        
        with error_context("prediction generation", zip_code):
            logger.info(f"Generating predictions for ZIP {zip_code}")
            
            # Get historical data
            time_series = self.data_processor.get_zip_time_series(zip_code)
            if time_series is None or time_series.empty:
                raise ValidationError(f"No data available for ZIP code {zip_code}")
            
            # Get metadata
            metadata = self.data_processor.get_zip_metadata(zip_code)
            
            # Validate data quality
            if self.enable_validation:
                self._validate_time_series_quality(time_series)
            
            # Prepare base prediction results
            results = {
                "zip_code": zip_code,
                "metadata": metadata,
                "current_value": float(time_series.iloc[-1]),
                "data_points": len(time_series),
                "data_start": time_series.index[0].isoformat(),
                "data_end": time_series.index[-1].isoformat(),
                "predictions": [],
                "model_info": self.model.get_model_info(),
                "generated_at": datetime.now().isoformat()
            }
            
            # Generate predictions for each horizon
            for horizon in forecast_horizons:
                try:
                    prediction = self._generate_single_prediction(
                        time_series, horizon, confidence_level, num_samples
                    )
                    results["predictions"].append(prediction)
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for horizon {horizon}: {e}")
                    # Continue with other horizons
            
            if not results["predictions"]:
                raise RuntimeError("Failed to generate any predictions")
            
            # Add trend analysis if requested
            if include_trends:
                try:
                    results["trend_analysis"] = self._analyze_trends(time_series)
                except Exception as e:
                    logger.warning(f"Failed to analyze trends: {e}")
                    results["trend_analysis"] = {"error": str(e)}
            
            # Add seasonal analysis if requested
            if include_seasonality:
                try:
                    results["seasonal_analysis"] = self._analyze_seasonality(time_series)
                except Exception as e:
                    logger.warning(f"Failed to analyze seasonality: {e}")
                    results["seasonal_analysis"] = {"error": str(e)}
            
            # Add market insights
            try:
                results["market_insights"] = self._generate_market_insights(
                    time_series, results["predictions"]
                )
            except Exception as e:
                logger.warning(f"Failed to generate market insights: {e}")
                results["market_insights"] = {"error": str(e)}
            
            logger.info(f"Successfully generated predictions for ZIP {zip_code}")
            return results
    
    def _generate_single_prediction(
        self,
        time_series: pd.Series,
        horizon: int,
        confidence_level: float,
        num_samples: int
    ) -> Dict[str, Any]:
        """Generate prediction for a single forecast horizon."""
        
        # Generate forecast using the model
        forecast_result = self.model.predict(
            time_series=time_series.values,
            forecast_horizon=horizon,
            num_samples=num_samples
        )
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Extract forecast values
        mean_forecast = forecast_result["mean"]
        
        # Calculate prediction intervals
        samples = np.array(forecast_result["samples"])
        if samples.size == 0:
            raise RuntimeError("No forecast samples generated")
            
        lower_bound = np.percentile(samples, lower_percentile, axis=0)
        upper_bound = np.percentile(samples, upper_percentile, axis=0)
        
        # Current value for percentage calculations
        current_value = float(time_series.iloc[-1])
        
        # Final predicted value (end of horizon)
        final_predicted_value = mean_forecast[-1]
        
        # Calculate percentage change
        percentage_change = calculate_percentage_change(current_value, final_predicted_value)
        
        # Generate future dates
        last_date = time_series.index[-1]
        try:
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'  # Month start
            )
        except Exception as e:
            logger.warning(f"Failed to generate future dates: {e}")
            # Fallback to simple date generation
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]
        
        return {
            "horizon_months": horizon,
            "predicted_value": float(final_predicted_value),
            "predicted_value_formatted": format_currency(final_predicted_value),
            "percentage_change": round(percentage_change, 2),
            "confidence_level": confidence_level,
            "confidence_interval": {
                "lower": float(lower_bound[-1]),
                "upper": float(upper_bound[-1]),
                "lower_formatted": format_currency(lower_bound[-1]),
                "upper_formatted": format_currency(upper_bound[-1])
            },
            "forecast_path": {
                "dates": [date.isoformat() for date in future_dates],
                "values": [float(val) for val in mean_forecast],
                "lower_bounds": [float(val) for val in lower_bound],
                "upper_bounds": [float(val) for val in upper_bound]
            },
            "prediction_quality": self._assess_prediction_quality(
                time_series, forecast_result, horizon
            )
        }
    
    def _validate_time_series_quality(self, time_series: pd.Series) -> None:
        """Validate the quality of time series data."""
        
        # Check minimum data points
        if len(time_series) < MIN_HISTORY_POINTS:
            raise ValidationError(
                f"Insufficient data: {len(time_series)} points, "
                f"minimum required: {MIN_HISTORY_POINTS}"
            )
        
        # Check for too many missing values
        missing_ratio = time_series.isna().sum() / len(time_series)
        if missing_ratio > 0.3:
            raise ValidationError(
                f"Too many missing values: {missing_ratio:.1%} of data points"
            )
        
        # Check for data recency
        last_date = time_series.index[-1]
        if pd.isna(last_date):
            raise ValidationError("Invalid last date in time series")
            
        # Handle timezone-aware datetime comparison
        current_time = datetime.now()
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            # If last_date is timezone-aware, make current_time timezone-aware too
            import pytz
            current_time = current_time.replace(tzinfo=pytz.UTC)
        elif hasattr(last_date, 'tz'):
            # If last_date is timezone-naive, ensure current_time is also naive
            current_time = current_time.replace(tzinfo=None)
            
        months_since_last = (current_time - last_date).days / 30.44
        if months_since_last > 6:
            logger.warning(
                f"Data may be stale: last data point is {months_since_last:.1f} months old"
            )
        
        # Check for extreme outliers
        values = time_series.dropna()
        if len(values) == 0:
            raise ValidationError("No valid data points after removing NaN values")
            
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        if iqr > 0:  # Avoid division by zero
            outlier_threshold = 3 * iqr
            outliers = values[(values < q1 - outlier_threshold) | (values > q3 + outlier_threshold)]
            
            if len(outliers) > len(values) * 0.1:
                logger.warning(f"High number of outliers detected: {len(outliers)} points")
    
    def _analyze_trends(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze trends in the time series data."""
        
        values = time_series.dropna()
        
        if len(values) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate various trend metrics
        short_term_months = min(12, max(2, len(values) // 2))
        medium_term_months = min(24, max(2, int(len(values) // 1.5)))
        
        # Recent trends
        short_term_trend = self._calculate_trend(values.tail(short_term_months))
        medium_term_trend = self._calculate_trend(values.tail(medium_term_months))
        long_term_trend = self._calculate_trend(values)
        
        # Volatility analysis
        returns = values.pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(12)  # Annualized volatility
        else:
            volatility = 0.0
        
        # Growth rates
        yoy_growth = self._calculate_year_over_year_growth(values)
        
        return {
            "short_term_trend": {
                "months": short_term_months,
                "slope": float(short_term_trend),
                "direction": "increasing" if short_term_trend > 0 else "decreasing"
            },
            "medium_term_trend": {
                "months": medium_term_months,
                "slope": float(medium_term_trend),
                "direction": "increasing" if medium_term_trend > 0 else "decreasing"
            },
            "long_term_trend": {
                "months": len(values),
                "slope": float(long_term_trend),
                "direction": "increasing" if long_term_trend > 0 else "decreasing"
            },
            "volatility": {
                "annualized": float(volatility),
                "level": self._classify_volatility(volatility)
            },
            "year_over_year_growth": yoy_growth
        }
    
    def _analyze_seasonality(self, time_series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns in the time series."""
        
        values = time_series.dropna()
        
        if len(values) < 24:  # Need at least 2 years for seasonal analysis
            return {"insufficient_data": True}
        
        try:
            # Extract month from index
            monthly_data = values.groupby(values.index.month).agg(['mean', 'std', 'count'])
            
            # Calculate seasonal indices
            overall_mean = values.mean()
            if overall_mean == 0:
                return {"error": "Cannot calculate seasonal indices with zero mean"}
                
            seasonal_indices = {}
            
            for month in range(1, 13):
                if month in monthly_data.index:
                    month_mean = monthly_data.loc[month, 'mean']
                    seasonal_index = (month_mean / overall_mean - 1) * 100
                    seasonal_indices[month] = {
                        "index": float(seasonal_index),
                        "average_value": float(month_mean),
                        "observations": int(monthly_data.loc[month, 'count'])
                    }
            
            # Identify peak and trough months
            if seasonal_indices:
                peak_month = max(seasonal_indices.keys(), 
                               key=lambda x: seasonal_indices[x]["index"])
                trough_month = min(seasonal_indices.keys(), 
                                 key=lambda x: seasonal_indices[x]["index"])
            else:
                peak_month = trough_month = None
            
            return {
                "seasonal_indices": seasonal_indices,
                "peak_month": peak_month,
                "trough_month": trough_month,
                "seasonal_strength": self._calculate_seasonal_strength(values)
            }
        except Exception as e:
            logger.warning(f"Error in seasonal analysis: {e}")
            return {"error": str(e)}
    
    def _generate_market_insights(
        self, 
        time_series: pd.Series, 
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate market insights and investment recommendations."""
        
        current_value = float(time_series.iloc[-1])
        
        # Calculate historical performance metrics
        annual_return = None
        if len(time_series) >= 12:
            one_year_ago_value = time_series.iloc[-12]
            if not pd.isna(one_year_ago_value) and one_year_ago_value != 0:
                annual_return = calculate_percentage_change(one_year_ago_value, current_value)
        
        # Analyze prediction confidence
        prediction_confidence = self._analyze_prediction_confidence(predictions)
        
        # Investment recommendation
        recommendation = self._generate_investment_recommendation(
            time_series, predictions, prediction_confidence
        )
        
        # Risk assessment
        risk_assessment = self._assess_investment_risk(time_series, predictions)
        
        # Total return calculation
        total_return = None
        if len(time_series) > 1:
            first_value = time_series.iloc[0]
            if not pd.isna(first_value) and first_value != 0:
                total_return = calculate_percentage_change(first_value, current_value)
        
        return {
            "current_market_position": self._assess_market_position(time_series),
            "historical_performance": {
                "annual_return": annual_return,
                "total_return": total_return
            },
            "prediction_confidence": prediction_confidence,
            "investment_recommendation": recommendation,
            "risk_assessment": risk_assessment,
            "key_insights": self._generate_key_insights(time_series, predictions)
        }
    
    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values.values
        
        # Check for invalid values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = y[~(np.isnan(y) | np.isinf(y))]
            x = x[:len(y)]
            
        if len(x) < 2:
            return 0.0
        
        # Simple linear regression with division by zero protection
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)
        
        denominator = n * sum_x2 - sum_x**2
        if abs(denominator) < 1e-10:  # Avoid division by zero
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        return float(slope)
    
    def _calculate_year_over_year_growth(self, values: pd.Series) -> List[Dict[str, Any]]:
        """Calculate year-over-year growth rates."""
        yoy_growth = []
        
        if len(values) < 12:
            return yoy_growth
        
        for i in range(12, len(values)):
            current_value = values.iloc[i]
            previous_year_value = values.iloc[i-12]
            
            if pd.isna(current_value) or pd.isna(previous_year_value) or previous_year_value == 0:
                continue
                
            growth_rate = calculate_percentage_change(previous_year_value, current_value)
            
            yoy_growth.append({
                "date": values.index[i].isoformat(),
                "growth_rate": round(growth_rate, 2)
            })
        
        return yoy_growth[-12:] if yoy_growth else []  # Return last 12 months
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level."""
        if volatility < 0.05:
            return "low"
        elif volatility < 0.15:
            return "moderate"
        else:
            return "high"
    
    def _calculate_seasonal_strength(self, values: pd.Series) -> float:
        """Calculate the strength of seasonal patterns."""
        if len(values) < 24:
            return 0.0
        
        try:
            # Simple seasonal strength calculation
            monthly_means = values.groupby(values.index.month).mean()
            overall_mean = values.mean()
            
            if overall_mean == 0:
                return 0.0
            
            seasonal_variance = np.var(monthly_means)
            total_variance = np.var(values)
            
            if total_variance == 0:
                return 0.0
                
            return float(seasonal_variance / total_variance)
        except Exception:
            return 0.0
    
    def _assess_prediction_quality(
        self, 
        time_series: pd.Series, 
        forecast_result: Dict[str, Any], 
        horizon: int
    ) -> Dict[str, Any]:
        """Assess the quality and reliability of predictions."""
        
        # Data quality score
        data_quality = min(1.0, len(time_series) / (MIN_HISTORY_POINTS * 2))
        
        # Forecast uncertainty (based on confidence interval width)
        samples = np.array(forecast_result["samples"])
        if samples.size == 0:
            uncertainty = 1.0  # Maximum uncertainty if no samples
        else:
            final_predictions = samples[:, -1]  # Final horizon predictions
            mean_pred = np.mean(final_predictions)
            if mean_pred == 0:
                uncertainty = 1.0
            else:
                uncertainty = np.std(final_predictions) / abs(mean_pred)
        
        # Horizon penalty (longer horizons are less reliable)
        horizon_penalty = max(0.5, 1.0 - (horizon - 3) * 0.1)
        
        # Overall quality score
        quality_score = data_quality * horizon_penalty * (1 - min(uncertainty, 0.5))
        
        return {
            "quality_score": round(float(quality_score), 3),
            "data_quality": round(float(data_quality), 3),
            "uncertainty": round(float(uncertainty), 3),
            "horizon_penalty": round(float(horizon_penalty), 3),
            "reliability": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low"
        }
    
    def _assess_market_position(self, time_series: pd.Series) -> str:
        """Assess current market position relative to historical values."""
        current_value = time_series.iloc[-1]
        historical_values = time_series.iloc[:-1]
        
        if len(historical_values) == 0:
            return "unknown"
        
        # Remove NaN values for comparison
        valid_historical = historical_values.dropna()
        if len(valid_historical) == 0:
            return "unknown"
        
        percentile = (valid_historical < current_value).mean() * 100
        
        if percentile >= 90:
            return "near_peak"
        elif percentile >= 70:
            return "above_average"
        elif percentile >= 30:
            return "average"
        elif percentile >= 10:
            return "below_average"
        else:
            return "near_bottom"
    
    def _analyze_prediction_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall confidence in predictions."""
        
        if not predictions:
            return {"overall_confidence": "low", "reason": "no_predictions"}
        
        # Average quality scores
        quality_scores = []
        for pred in predictions:
            if "prediction_quality" in pred and "quality_score" in pred["prediction_quality"]:
                quality_scores.append(pred["prediction_quality"]["quality_score"])
        
        if not quality_scores:
            return {"overall_confidence": "low", "reason": "no_quality_scores"}
        
        avg_quality = np.mean(quality_scores)
        
        # Consistency across horizons
        percentage_changes = []
        for pred in predictions:
            if "percentage_change" in pred:
                percentage_changes.append(pred["percentage_change"])
        
        if len(percentage_changes) > 1:
            consistency = max(0.0, 1.0 - (np.std(percentage_changes) / 100))  # Normalize
        else:
            consistency = 1.0  # Perfect consistency if only one prediction
        
        # Overall confidence
        overall_confidence = (avg_quality + consistency) / 2
        
        confidence_level = "high" if overall_confidence > 0.7 else "medium" if overall_confidence > 0.4 else "low"
        
        return {
            "overall_confidence": confidence_level,
            "average_quality": round(float(avg_quality), 3),
            "consistency": round(float(consistency), 3),
            "confidence_score": round(float(overall_confidence), 3)
        }
    
    def _generate_investment_recommendation(
        self, 
        time_series: pd.Series, 
        predictions: List[Dict[str, Any]], 
        confidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment recommendation based on analysis."""
        
        if not predictions:
            return {"recommendation": "hold", "reason": "insufficient_predictions"}
        
        # Get short and medium term predictions
        short_term = next((p for p in predictions if p["horizon_months"] <= 6), None)
        medium_term = next((p for p in predictions if p["horizon_months"] >= 12), None)
        
        if not short_term:
            return {"recommendation": "hold", "reason": "no_short_term_prediction"}
        
        short_term_change = short_term.get("percentage_change", 0)
        medium_term_change = medium_term.get("percentage_change", short_term_change) if medium_term else short_term_change
        
        # Decision logic
        if confidence.get("overall_confidence") == "low":
            recommendation = "hold"
            reason = "low_prediction_confidence"
        elif short_term_change > 5 and medium_term_change > 3:
            recommendation = "buy"
            reason = "strong_growth_expected"
        elif short_term_change < -5 and medium_term_change < -3:
            recommendation = "sell"
            reason = "decline_expected"
        elif short_term_change > 2:
            recommendation = "buy"
            reason = "moderate_growth_expected"
        elif short_term_change < -2:
            recommendation = "hold"
            reason = "potential_decline"
        else:
            recommendation = "hold"
            reason = "stable_market"
        
        return {
            "recommendation": recommendation,
            "reason": reason,
            "confidence": confidence.get("overall_confidence", "unknown"),
            "expected_return": {
                "short_term": short_term_change,
                "medium_term": medium_term_change
            }
        }
    
    def _assess_investment_risk(
        self, 
        time_series: pd.Series, 
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess investment risk based on historical data and predictions."""
        
        # Historical volatility
        returns = time_series.pct_change().dropna()
        if len(returns) > 0:
            historical_volatility = returns.std() * np.sqrt(12)  # Annualized
        else:
            historical_volatility = 0.0
        
        # Prediction uncertainty
        if predictions:
            uncertainties = []
            for pred in predictions:
                ci = pred.get("confidence_interval", {})
                predicted_value = pred.get("predicted_value", 0)
                
                if ci and predicted_value != 0:
                    uncertainty = (ci.get("upper", 0) - ci.get("lower", 0)) / abs(predicted_value)
                    uncertainties.append(uncertainty)
            
            avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
        else:
            avg_uncertainty = 0.5
        
        # Risk classification
        risk_score = (historical_volatility + avg_uncertainty) / 2
        
        if risk_score < 0.1:
            risk_level = "low"
        elif risk_score < 0.2:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "risk_score": round(float(risk_score), 3),
            "historical_volatility": round(float(historical_volatility), 3),
            "prediction_uncertainty": round(float(avg_uncertainty), 3)
        }
    
    def _generate_key_insights(
        self, 
        time_series: pd.Series, 
        predictions: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate key insights and takeaways."""
        
        insights = []
        
        # Data insights
        insights.append(f"Based on {len(time_series)} months of historical data")
        
        # Current market position
        market_position = self._assess_market_position(time_series)
        position_descriptions = {
            "near_peak": "Current prices are near historical peaks",
            "above_average": "Current prices are above historical average",
            "average": "Current prices are around historical average",
            "below_average": "Current prices are below historical average",
            "near_bottom": "Current prices are near historical lows",
            "unknown": "Market position unclear due to insufficient data"
        }
        insights.append(position_descriptions.get(market_position, "Market position unclear"))
        
        # Prediction insights
        if predictions:
            short_term = next((p for p in predictions if p["horizon_months"] <= 6), None)
            if short_term:
                change = short_term.get("percentage_change", 0)
                if abs(change) > 5:
                    direction = "increase" if change > 0 else "decrease"
                    insights.append(f"Significant {direction} expected in next 6 months ({change:+.1f}%)")
                else:
                    insights.append("Relatively stable prices expected in the short term")
        
        # Trend insights
        if len(time_series) >= 12:
            try:
                recent_trend = self._calculate_trend(time_series.tail(12))
                time_series_std = time_series.std()
                if not pd.isna(time_series_std) and time_series_std > 0:
                    if abs(recent_trend) > time_series_std / 12:
                        direction = "upward" if recent_trend > 0 else "downward"
                        insights.append(f"Recent 12-month trend shows {direction} movement")
            except Exception:
                pass  # Skip trend insight if calculation fails
        
        return insights


def quick_predict(
    zip_code: str,
    forecast_months: Optional[List[int]] = None,
    data_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick prediction function for simple use cases.
    
    Args:
        zip_code: 5-digit ZIP code
        forecast_months: List of forecast horizons
        data_path: Path to ZHVI data file
        
    Returns:
        Prediction results
        
    Raises:
        ValidationError: If input validation fails
        RuntimeError: If prediction fails
    """
    predictor = HomePricePredictor(data_path=data_path)
    return predictor.predict_zip_price(
        zip_code=zip_code,
        forecast_horizons=forecast_months
    )


def batch_predict(
    zip_codes: List[str],
    forecast_horizons: Optional[List[int]] = None,
    data_path: Optional[str] = None,
    continue_on_error: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Batch prediction for multiple ZIP codes.
    
    Args:
        zip_codes: List of ZIP codes
        forecast_horizons: List of forecast horizons
        data_path: Path to ZHVI data file
        continue_on_error: Whether to continue processing other ZIP codes if one fails
        
    Returns:
        Dictionary mapping ZIP codes to prediction results
    """
    if not zip_codes:
        return {}
    
    predictor = HomePricePredictor(data_path=data_path)
    predictor.initialize()  # Initialize once for efficiency
    
    results = {}
    for zip_code in zip_codes:
        try:
            results[zip_code] = predictor.predict_zip_price(
                zip_code=zip_code,
                forecast_horizons=forecast_horizons
            )
        except Exception as e:
            logger.error(f"Failed to predict for ZIP {zip_code}: {e}")
            if continue_on_error:
                results[zip_code] = {"error": str(e)}
            else:
                raise
    
    return results 