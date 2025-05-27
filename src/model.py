"""
Chronos T5 Model Wrapper for Home Price Prediction

This module provides a wrapper around the Chronos T5 Small model for time series forecasting
of home prices. It handles model loading, input preprocessing, inference, and output formatting.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chronos import ChronosPipeline

logger = logging.getLogger(__name__)


class ChronosT5Model:
    """
    Wrapper class for Chronos T5 Small model for home price forecasting.
    
    This class handles:
    - Model loading and caching
    - Input preprocessing and validation
    - Inference with proper context length
    - Output formatting with confidence intervals
    """
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Chronos T5 model.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache the model
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "./data/model_cache/"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Model configuration
        self.max_context_length = 512  # Chronos T5 Small context length
        self.min_history_points = 12   # Minimum data points for reliable prediction
        
        logger.info(f"Initializing Chronos T5 model on device: {self.device}")
        
    def load_model(self) -> None:
        """Load the Chronos T5 model and tokenizer."""
        try:
            logger.info(f"Loading Chronos model: {self.model_name}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load the Chronos pipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir
            )
            
            logger.info("Chronos model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {str(e)}")
            raise
    
    def preprocess_time_series(
        self,
        time_series: Union[List[float], np.ndarray, pd.Series],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess time series data for Chronos model input.
        
        Args:
            time_series: Input time series data
            max_length: Maximum length to truncate the series
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to numpy array
        if isinstance(time_series, pd.Series):
            ts_array = time_series.values
        elif isinstance(time_series, list):
            ts_array = np.array(time_series)
        else:
            ts_array = time_series
        
        # Remove NaN values - ensure we have float64 array
        ts_array = ts_array.astype(np.float64)
        ts_array = ts_array[~np.isnan(ts_array)]
        
        if len(ts_array) < self.min_history_points:
            raise ValueError(
                f"Time series too short. Need at least {self.min_history_points} "
                f"data points, got {len(ts_array)}"
            )
        
        # Truncate to max context length if needed
        max_len = max_length or self.max_context_length
        if len(ts_array) > max_len:
            ts_array = ts_array[-max_len:]
            logger.warning(f"Truncated time series to last {max_len} points")
        
        # Convert to tensor
        return torch.tensor(ts_array, dtype=torch.float32)
    
    def predict(
        self,
        time_series: Union[List[float], np.ndarray, pd.Series],
        forecast_horizon: int,
        num_samples: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Generate forecasts for the given time series.
        
        Args:
            time_series: Historical time series data
            forecast_horizon: Number of periods to forecast
            num_samples: Number of sample paths to generate
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Dictionary containing forecast statistics
        """
        if self.pipeline is None:
            self.load_model()
        
        try:
            # Preprocess input
            context = self.preprocess_time_series(time_series)
            
            logger.info(
                f"Generating forecast for {forecast_horizon} periods "
                f"with {len(context)} historical points"
            )
            
            # Generate forecast
            forecast = self.pipeline.predict(
                context=context.unsqueeze(0),  # Add batch dimension
                prediction_length=forecast_horizon,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Extract forecast samples (remove batch dimension)
            forecast_samples = forecast[0].numpy()  # Shape: (num_samples, forecast_horizon)
            
            # Calculate statistics
            mean_forecast = np.mean(forecast_samples, axis=0)
            median_forecast = np.median(forecast_samples, axis=0)
            std_forecast = np.std(forecast_samples, axis=0)
            
            # Calculate confidence intervals
            percentiles = [10, 25, 75, 90]
            confidence_intervals = {}
            for p in percentiles:
                confidence_intervals[f"p{p}"] = np.percentile(forecast_samples, p, axis=0)
            
            return {
                "mean": mean_forecast.tolist(),
                "median": median_forecast.tolist(),
                "std": std_forecast.tolist(),
                "samples": forecast_samples.tolist(),
                "confidence_intervals": confidence_intervals,
                "forecast_horizon": forecast_horizon,
                "num_samples": num_samples
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_single_value(
        self,
        time_series: Union[List[float], np.ndarray, pd.Series],
        forecast_horizon: int,
        confidence_level: float = 0.8
    ) -> Dict[str, float]:
        """
        Generate a single point forecast with confidence interval.
        
        Args:
            time_series: Historical time series data
            forecast_horizon: Number of periods to forecast
            confidence_level: Confidence level for interval (e.g., 0.8 for 80%)
            
        Returns:
            Dictionary with point forecast and confidence interval
        """
        # Get full forecast
        forecast_result = self.predict(time_series, forecast_horizon)
        
        # Extract final period forecast
        final_mean = forecast_result["mean"][-1]
        final_std = forecast_result["std"][-1]
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        samples = np.array(forecast_result["samples"])[:, -1]  # Final period samples
        lower_bound = np.percentile(samples, lower_percentile)
        upper_bound = np.percentile(samples, upper_percentile)
        
        return {
            "predicted_value": final_mean,
            "confidence_interval": [lower_bound, upper_bound],
            "confidence_level": confidence_level,
            "std": final_std
        }
    
    def batch_predict(
        self,
        time_series_list: List[Union[List[float], np.ndarray, pd.Series]],
        forecast_horizons: List[int],
        **kwargs
    ) -> List[Dict]:
        """
        Generate forecasts for multiple time series.
        
        Args:
            time_series_list: List of time series to forecast
            forecast_horizons: List of forecast horizons for each series
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            List of forecast results
        """
        if len(time_series_list) != len(forecast_horizons):
            raise ValueError("Number of time series must match number of forecast horizons")
        
        results = []
        for i, (ts, horizon) in enumerate(zip(time_series_list, forecast_horizons)):
            try:
                logger.info(f"Processing batch item {i+1}/{len(time_series_list)}")
                result = self.predict(ts, horizon, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch item {i+1}: {str(e)}")
                results.append({"error": str(e)})
        
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "max_context_length": self.max_context_length,
            "min_history_points": self.min_history_points,
            "is_loaded": self.pipeline is not None
        }


# Convenience function for quick predictions
def quick_forecast(
    time_series: Union[List[float], np.ndarray, pd.Series],
    forecast_months: int,
    model_cache_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Quick forecast function for simple use cases.
    
    Args:
        time_series: Historical home price data
        forecast_months: Number of months to forecast
        model_cache_dir: Directory to cache the model
        
    Returns:
        Simple forecast result with prediction and confidence interval
    """
    model = ChronosT5Model(cache_dir=model_cache_dir)
    return model.predict_single_value(time_series, forecast_months)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate sample time series (simulating monthly home prices)
    np.random.seed(42)
    months = 60
    trend = np.linspace(500000, 600000, months)
    seasonal = 10000 * np.sin(2 * np.pi * np.arange(months) / 12)
    noise = np.random.normal(0, 5000, months)
    sample_prices = trend + seasonal + noise
    
    # Initialize model
    model = ChronosT5Model()
    
    # Generate forecast
    forecast_result = model.predict_single_value(
        time_series=sample_prices,
        forecast_horizon=6,  # 6 months ahead
        confidence_level=0.8
    )
    
    print("Forecast Result:")
    print(f"Predicted Value: ${forecast_result['predicted_value']:,.2f}")
    print(f"Confidence Interval: ${forecast_result['confidence_interval'][0]:,.2f} - ${forecast_result['confidence_interval'][1]:,.2f}")
    print(f"Standard Deviation: ${forecast_result['std']:,.2f}") 