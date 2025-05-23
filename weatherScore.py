import numpy as np

def calculate_weather_score(temperature, humidity, precipitation, wind_speed, sunshine_hours,
                           temp_opt=20, temp_sigma=5,
                           weights=(0.05, 0.45, 0.1, 0.1, 0.3)):
    """
    Calculate weather score for car rental revenue prediction
    
    Parameters:
    temperature (float): Average temperature in °C
    humidity (float): Average relative humidity in %
    precipitation (float): Precipitation in mm
    wind_speed (float): Average wind speed in m/s
    sunshine_hours (float): Daily sunshine duration in hours
    temp_opt (float): Optimal temperature (default 20°C)
    temp_sigma (float): Temperature standard deviation (default 5)
    weights (tuple): (temp, humidity, precip, wind, sunshine) weights
    
    Returns:
    float: Weather score between 0-1
    """
    # Temperature component (Gaussian bell curve)
    temp_score = np.exp(-(temperature - temp_opt)**2 / (2 * temp_sigma**2))
    
    # Humidity component (linear decrease)
    humidity_score = max(0, 1 - humidity/100)
    
    # Precipitation component (hyperbolic tangent decay)
    precip_score = 1 - np.tanh(precipitation/10)
    
    # Wind speed component (linear decrease)
    wind_score = max(0, 1 - wind_speed/20)
    
    # Sunshine component (linear increase)
    sunshine_score = min(1, sunshine_hours/12)
    
    # Weighted sum
    w_temp, w_humid, w_precip, w_wind, w_sun = weights
    total_score = (
        w_temp * temp_score +
        w_humid * humidity_score +
        w_precip * precip_score +
        w_wind * wind_score +
        w_sun * sunshine_score
    )
    
    return max(0, min(1, total_score))

if __name__ == '__main__':
    # Example usage
    T = 25.0  # Temperature in °C
    RH = 60.0  # Relative Humidity in %
    P = 5.0   # Precipitation in mm
    W = 3.0   # Wind Speed in m/s
    S = 8.0   # Sunshine Hours in hours
    
    score = calculate_weather_score(T, RH, P, W, S)
    print(f"Weather Score: {score:.2f} / 1")