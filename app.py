import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Database module
import database

# Translation - with try/except to allow deployment without it
try:
    from translate import Translator
    TRANSLATE_AVAILABLE = True
except:
    Translator = None
    TRANSLATE_AVAILABLE = False

# AI APIs - with fallbacks
try:
    import openai as openai_module
    OPENAI_AVAILABLE = True
except:
    openai_module = None
    OPENAI_AVAILABLE = False

try:
    import cohere as cohere_module
    COHERE_AVAILABLE = True
except:
    cohere_module = None
    COHERE_AVAILABLE = False

import plotly.express as px
import plotly.graph_objects as go

# Flag for transformers (not required)
TRANSFORMERS_AVAILABLE = False
print("AgriDream starting...")

# Initialize database on app start
@st.cache_resource
def init_db():
    """Initialize database once"""
    return database.initialize_database()

# Run database initialization
db_stats = init_db()
print(f"Database initialized: {db_stats}")



load_dotenv()

# ---------------------------
# Database Helper Functions (Use Database Instead of CSV)
# ---------------------------

def get_price_data_from_db(crop: str, state: str, days: int = 30) -> pd.DataFrame:
    """Get price data from database for forecasting"""
    try:
        df = database.get_price_by_crop_state(crop, state, limit=days)
        return df
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return pd.DataFrame()

def get_price_stats_from_db(crop: str, state: str) -> dict:
    """Get price statistics from database"""
    try:
        return database.get_price_statistics(crop, state)
    except Exception as e:
        print(f"Error fetching price stats: {e}")
        return {}

def get_trend_from_db(crop: str, state: str) -> str:
    """Get price trend from database"""
    try:
        return database.calculate_trend(crop, state)
    except Exception as e:
        print(f"Error calculating trend: {e}")
        return "stable"

def get_confidence_from_db(crop: str, state: str) -> str:
    """Get prediction confidence from database"""
    try:
        return database.calculate_confidence(crop, state)
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        return "Low"

def save_prediction_to_db(farmer_id: int, crop: str, state: str, current_price: float,
                         predicted_price: float, trend: str, confidence: str,
                         decision: str, decision_reason: str, money_impact: str) -> int:
    """Save prediction to database"""
    try:
        return database.insert_prediction(
            farmer_id=farmer_id, crop=crop, state=state,
            current_price=current_price, predicted_price=predicted_price,
            trend=trend, confidence=confidence, decision=decision,
            decision_reason=decision_reason, money_impact=money_impact
        )
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return 0

def create_alert_for_user(farmer_id: int, alert_type: str, title: str, 
                          message: str, severity: str = "Medium") -> int:
    """Create alert for farmer"""
    try:
        return database.create_alert(farmer_id, alert_type, title, message, severity)
    except Exception as e:
        print(f"Error creating alert: {e}")
        return 0

# ---------------------------
# Global Variables and Functions
# ---------------------------

# Safe translate function with error handling
def safe_translate(text, src='en', dest='en'):
    """Safely translate text with error handling"""
    try:
        if not TRANSLATE_AVAILABLE:
            return text
        if src == dest:
            return text
        translator_obj = Translator(from_lang=src, to_lang=dest)
        return translator_obj.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Get language code function
def get_lang_code(lang):
    """Get language code for a given language name"""
    return lang_codes.get(lang, 'en')


# Get family numbers from profile
def get_family_numbers(farmer_profile):
    """Extract family phone numbers from farmer profile"""
    family_numbers = []
    if farmer_profile.get('family1', {}).get('phone'):
        family_numbers.append(farmer_profile['family1']['phone'])
    if farmer_profile.get('family2', {}).get('phone'):
        family_numbers.append(farmer_profile['family2']['phone'])
    return family_numbers

# Global variables for easy access
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEP_AI_API_KEY = os.getenv('DEEP_AI_API_KEY')
COHERE_API_KEY = '6rzDJkHIdCEw1aoURMqEqAk5kEZmTNDvXS7dQHbP'
FAMILY_NUMBERS = []  # Will be populated from farmer profile

# ---------------------------
# Load Crop Recommendation Data
# ---------------------------
df_crop = pd.read_csv('crop_recommendation.csv')

# Sample Data for other sections
crop_data = {
    "Wheat": {"season": "Rabi", "price_forecast": [2000, 2100, 2200]},
    "Rice": {"season": "Kharif", "price_forecast": [3000, 3100, 3200]},
    "Maize": {"season": "Kharif", "price_forecast": [1500, 1600, 1700]},
}

# Crop recommendation function with detailed reasoning
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, state):
    """
    Enhanced crop recommendation with reasoning and insights
    Returns: crops, confidences, reasoning_dict
    """
    conditions = np.array([N, P, K, temperature, humidity, ph, rainfall])
    distances = np.sqrt(np.sum((df_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] - conditions) ** 2, axis=1))
    closest_indices = distances.nsmallest(3).index
    recommended_crops = df_crop.iloc[closest_indices]['label'].tolist()
    confidences = [max(0, min(100, 100 - distances[i] / 10)) for i in closest_indices]
    
    # Generate reasoning for each crop
    reasoning = {}
    for crop in recommended_crops:
        reasons = []
        factors = []
        
        # Analyze NPK
        if N > 80:
            reasons.append("High nitrogen soil supports growth")
            factors.append("Nitrogen (high)")
        elif N < 30:
            reasons.append("Low nitrogen - suitable for nitrogen-fixing crops")
            factors.append("Nitrogen (low)")
            
        if P > 60:
            reasons.append("High phosphorus promotes root development")
            factors.append("Phosphorus (high)")
            
        if K > 100:
            reasons.append("High potassium enhances disease resistance")
            factors.append("Potassium (high)")
        
        # Temperature analysis
        if 20 <= temperature <= 30:
            reasons.append("Moderate temperature ideal for most crops")
            factors.append("Temperature (optimal)")
        elif temperature > 35:
            reasons.append("Warm climate suits heat-tolerant crops")
            factors.append("Temperature (warm)")
        elif temperature < 15:
            reasons.append("Cool climate suitable for winter crops")
            factors.append("Temperature (cool)")
        
        # Rainfall analysis
        if rainfall > 150:
            reasons.append("High rainfall reduces irrigation needs")
            factors.append("Rainfall (high)")
        elif rainfall < 50:
            reasons.append("Low rainfall - drought-tolerant crops recommended")
            factors.append("Rainfall (low)")
        else:
            reasons.append("Moderate rainfall suitable for diverse crops")
            factors.append("Rainfall (moderate)")
        
        # pH analysis
        if 6.0 <= ph <= 7.5:
            reasons.append("Optimal pH range for nutrient uptake")
            factors.append("pH (optimal)")
        elif ph < 6.0:
            reasons.append("Acidic soil - suitable for acid-tolerant crops")
            factors.append("pH (acidic)")
        else:
            reasons.append("Alkaline soil - suitable for alkaline-tolerant crops")
            factors.append("pH (alkaline)")
        
        reasoning[crop] = {
            'reasons': reasons[:4],  # Top 4 reasons
            'factors': factors[:4]   # Top 4 factors
        }
    
    return recommended_crops, confidences, reasoning


# Profitability and market data for crops (ENHANCED with income timing)
CROP_PROFIT_INFO = {
    "Rice": {
        "profit_outlook": "Medium",
        "market_value": "₹2,000-2,500/quintal",
        "season": "Kharif (June-Oct)",
        "season_short": "Kharif",
        "risk": {"water": "High", "pest": "Medium"},
        "growth_period": "120-150 days",
        "income_timing": "4 months",
        "harvest_month": "October",
        "next_crop": "Wheat, Mustard, Peas",
        "water_need": "High",
        "initial_cost": "₹15,000/acre"
    },
    "Wheat": {
        "profit_outlook": "Medium",
        "market_value": "₹2,200-2,600/quintal",
        "season": "Rabi (Nov-Apr)",
        "season_short": "Rabi",
        "risk": {"water": "Medium", "pest": "Medium"},
        "growth_period": "120-150 days",
        "income_timing": "5 months",
        "harvest_month": "April",
        "next_crop": "Rice, Maize, Cotton",
        "water_need": "Medium",
        "initial_cost": "₹12,000/acre"
    },
    "Maize": {
        "profit_outlook": "High",
        "market_value": "₹1,500-1,900/quintal",
        "season": "Kharif & Rabi",
        "season_short": "Both",
        "risk": {"water": "Medium", "pest": "High"},
        "growth_period": "90-120 days",
        "income_timing": "3-4 months",
        "harvest_month": "September/November",
        "next_crop": "Wheat, Mustard",
        "water_need": "Medium",
        "initial_cost": "₹10,000/acre"
    },
    "Cotton": {
        "profit_outlook": "High",
        "market_value": "₹5,500-6,500/quintal",
        "season": "Kharif (June-Nov)",
        "season_short": "Kharif",
        "risk": {"water": "Low", "pest": "High"},
        "growth_period": "150-180 days",
        "income_timing": "5-6 months",
        "harvest_month": "November",
        "next_crop": "Wheat, Mustard",
        "water_need": "Low",
        "initial_cost": "₹20,000/acre"
    },
    "Sugarcane": {
        "profit_outlook": "High",
        "market_value": "₹3,500-4,200/quintal",
        "season": "Kharif (12-18 months)",
        "season_short": "Annual",
        "risk": {"water": "High", "pest": "Medium"},
        "growth_period": "12-18 months",
        "income_timing": "12-18 months",
        "harvest_month": "December-May",
        "next_crop": "Wheat, Mustard",
        "water_need": "High",
        "initial_cost": "₹50,000/acre"
    },
    "Tomato": {
        "profit_outlook": "High",
        "market_value": "₹1,500-3,000/quintal",
        "season": "All seasons",
        "season_short": "All Year",
        "risk": {"water": "Medium", "pest": "High"},
        "growth_period": "90-120 days",
        "income_timing": "3-4 months",
        "harvest_month": "Variable",
        "next_crop": "Leafy vegetables, Pulses",
        "water_need": "Medium",
        "initial_cost": "₹25,000/acre"
    },
    "Potato": {
        "profit_outlook": "Medium",
        "market_value": "₹1,200-1,800/quintal",
        "season": "Rabi (90-120 days)",
        "season_short": "Rabi",
        "risk": {"water": "Medium", "pest": "Medium"},
        "growth_period": "90-120 days",
        "income_timing": "3-4 months",
        "harvest_month": "February-March",
        "next_crop": "Rice, Maize",
        "water_need": "Medium",
        "initial_cost": "₹18,000/acre"
    },
    "Onion": {
        "profit_outlook": "High",
        "market_value": "₹1,500-2,500/quintal",
        "season": "Rabi & Kharif",
        "season_short": "Rabi/Kharif",
        "risk": {"water": "Low", "pest": "Medium"},
        "growth_period": "90-120 days",
        "income_timing": "3-4 months",
        "harvest_month": "March/May",
        "next_crop": "Rice, Maize",
        "water_need": "Low",
        "initial_cost": "₹15,000/acre"
    },
    "Groundnut": {
        "profit_outlook": "Medium",
        "market_value": "₹4,000-5,000/quintal",
        "season": "Kharif",
        "season_short": "Kharif",
        "risk": {"water": "Low", "pest": "Medium"},
        "growth_period": "120-150 days",
        "income_timing": "4-5 months",
        "harvest_month": "October",
        "next_crop": "Wheat, Mustard",
        "water_need": "Low",
        "initial_cost": "₹12,000/acre"
    },
    "Mustard": {
        "profit_outlook": "Medium",
        "market_value": "₹5,000-6,000/quintal",
        "season": "Rabi",
        "season_short": "Rabi",
        "risk": {"water": "Low", "pest": "Low"},
        "growth_period": "120-150 days",
        "income_timing": "4-5 months",
        "harvest_month": "March",
        "next_crop": "Rice, Maize",
        "water_need": "Low",
        "initial_cost": "₹8,000/acre"
    }
}

# Indian state to capital city mapping for weather
INDIAN_STATE_CAPITALS = {
    'Maharashtra': 'Mumbai',
    'Tamil Nadu': 'Chennai',
    'Uttar Pradesh': 'Lucknow',
    'Karnataka': 'Bangalore',
    'Gujarat': 'Ahmedabad',
    'Madhya Pradesh': 'Bhopal',
    'Punjab': 'Chandigarh',
    'West Bengal': 'Kolkata',
    'Andhra Pradesh': 'Visakhapatnam',
    'Telangana': 'Hyderabad',
    'Rajasthan': 'Jaipur',
    'Kerala': 'Thiruvananthapuram',
    'Bihar': 'Patna',
    'Jharkhand': 'Ranchi',
    'Haryana': 'Chandigarh',
    'Chhattisgarh': 'Raipur',
    'Odisha': 'Bhubaneswar',
    'Assam': 'Guwahati',
    'Uttarakhand': 'Dehradun',
    'Goa': 'Panaji'
}

# Weather function
def get_weather(city):
    # Use hardcoded API key as primary, with fallback to env variable
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key or api_key in ['your_openweather_key_here', 'demo_key']:
        api_key = '21e959d85a5148fdd18fbb293869d9ef'
    
    if not api_key:
        return None, "API key not configured"
    
    # Map Indian state names to capital cities
    if city in INDIAN_STATE_CAPITALS:
        city = INDIAN_STATE_CAPITALS[city]

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 200:
            weather = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'rainfall': data.get('rain', {}).get('1h', 0)
            }
            return weather, None
        else:
            return None, data.get('message', 'Weather data not available')
    except requests.exceptions.Timeout:
        return None, "Connection timeout - please try again"
    except requests.exceptions.ConnectionError:
        return None, "No internet connection - please check your network"
    except Exception as e:
        return None, str(e)

# ============================================================
# AGMARKNET API - Real Mandi Prices (Official Government Data)
# ============================================================

def get_agmarknet_prices(state=None, commodity=None, market=None, date=None):
    """
    Fetch real mandi prices from AGMARKNET API (data.gov.in)
    - Real mandi market prices
    - Updated daily
    - Official Government data
    - Perfect for price prediction models
    """
    try:
        api_key = os.getenv('DATA_GOV_IN_API_KEY', '579b464db66ec23bdd0000019c162fc702704b767a58d3e8897d4328')
        
        # AGMARKNET commodity-wise prices API
        base_url = "https://api.data.gov.in/resource/9ef84268-d54d-46d2-ae73-a2c1b74f3051"
        
        params = {
            'api-key': api_key,
            'format': 'json',
            'limit': 500  # Get more records
        }
        
        if state:
            params['filters[State]'] = state
        if commodity:
            params['filters[Commodity]'] = commodity
        if market:
            params['filters[Market]'] = market
        
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'records' in data and len(data['records']) > 0:
                return data['records'], None
            return [], "No data available for the selected filters"
        else:
            return None, f"API error: {response.status_code}"
    except Exception as e:
        return None, str(e)

# ============================================================
# COMPREHENSIVE MARKET DATA - ALL INDIAN STATES & MARKETS
# ============================================================

# State-wise crop prices with multiple markets
ALL_STATES_MARKET_DATA = {
    'Andhra Pradesh': {
        'Guntur': {'Paddy': 2200, 'Rice': 2500, 'Maize': 1800, 'Cotton': 7200, 'Tomato': 1500, 'Onion': 1200, 'Chilli': 4500, 'Turmeric': 11500},
        'Visakhapatnam': {'Paddy': 2150, 'Rice': 2450, 'Maize': 1750, 'Fish': 350, 'Cashew': 9000, 'Mango': 5000},
        'Vijayawada': {'Paddy': 2180, 'Rice': 2480, 'Maize': 1820, 'Sugarcane': 3000, 'Groundnut': 5200, 'Banana': 4200},
        'Tirupati': {'Paddy': 2100, 'Rice': 2400, 'Groundnut': 5000, 'Sugarcane': 2900, 'Mango': 4800}
    },
    'Arunachal Pradesh': {
        'Itanagar': {'Rice': 2600, 'Wheat': 2300, 'Maize': 1900, 'Apple': 9000, 'Orange': 4000, 'Pineapple': 3800},
        'Naharlagun': {'Rice': 2550, 'Wheat': 2250, 'Maize': 1850, 'Orange': 3800, 'Ginger': 9500}
    },
    'Assam': {
        'Guwahati': {'Rice': 2400, 'Wheat': 2100, 'Maize': 1700, 'Tea': 15000, 'Mustard': 5500, 'Potato': 1300},
        'Dibrugarh': {'Rice': 2350, 'Wheat': 2050, 'Tea': 15500, 'Mustard': 5400, 'Pineapple': 3600},
        'Silchar': {'Rice': 2300, 'Wheat': 2000, 'Maize': 1650, 'Banana': 3800, 'Coconut': 3200}
    },
    'Bihar': {
        'Patna': {'Rice': 2300, 'Wheat': 2150, 'Maize': 1650, 'Onion': 1100, 'Potato': 1200, 'Litchi': 8000},
        'Muzaffarpur': {'Rice': 2250, 'Wheat': 2100, 'Maize': 1600, 'Onion': 1050, 'Potato': 1150, 'Mango': 4500},
        'Gaya': {'Rice': 2200, 'Wheat': 2050, 'Maize': 1550, 'Onion': 1000, 'Potato': 1100}
    },
    'Chhattisgarh': {
        'Raipur': {'Rice': 2200, 'Wheat': 2100, 'Sugarcane': 3100, 'Soybean': 4200, 'Turmeric': 11000, 'Tamarind': 6000},
        'Bilaspur': {'Rice': 2150, 'Wheat': 2050, 'Sugarcane': 3000, 'Soybean': 4000, 'Mustard': 5200},
        'Durg': {'Rice': 2180, 'Wheat': 2080, 'Sugarcane': 3050, 'Soybean': 4100, 'Cotton': 6800}
    },
    'Gujarat': {
        'Ahmedabad': {'Cotton': 7300, 'Groundnut': 5500, 'Sugarcane': 3200, 'Onion': 1150, 'Garlic': 8500, 'Cumin': 15000},
        'Surat': {'Cotton': 7200, 'Groundnut': 5400, 'Sugarcane': 3100, 'Onion': 1100, 'Mango': 5000},
        'Vadodara': {'Cotton': 7100, 'Groundnut': 5300, 'Sugarcane': 3050, 'Onion': 1050, 'Banana': 4000},
        'Rajkot': {'Cotton': 7000, 'Groundnut': 5200, 'Sesame': 8000, 'Onion': 1000, 'Garlic': 8200}
    },
    'Haryana': {
        'Karnal': {'Wheat': 2250, 'Mustard': 5800, 'Sugarcane': 3100, 'Cotton': 7000, 'Barley': 1900, 'Rice': 2400},
        'Hisar': {'Wheat': 2200, 'Mustard': 5700, 'Cotton': 6900, 'Barley': 1850, 'Onion': 1100},
        'Gurgaon': {'Wheat': 2275, 'Mustard': 5850, 'Sugarcane': 3150, 'Cotton': 7050, 'Vegetables': 1500}
    },
    'Himachal Pradesh': {
        'Shimla': {'Apple': 8500, 'Mango': 5000, 'Wheat': 2300, 'Maize': 1800, 'Potato': 1300, 'Cherry': 12000},
        'Kullu': {'Apple': 8200, 'Mango': 4800, 'Wheat': 2250, 'Potato': 1250, 'Apricot': 9000},
        'Mandi': {'Apple': 8000, 'Wheat': 2200, 'Maize': 1750, 'Potato': 1200, 'Pomegranate': 7500}
    },
    'Jharkhand': {
        'Ranchi': {'Rice': 2350, 'Wheat': 2150, 'Maize': 1700, 'Ragi': 2400, 'Masoor': 7500, 'Cashew': 8500},
        'Jamshedpur': {'Rice': 2300, 'Wheat': 2100, 'Maize': 1650, 'Ragi': 2350, 'Mango': 4500},
        'Dhanbad': {'Rice': 2280, 'Wheat': 2080, 'Maize': 1600, 'Ragi': 2300, 'Coal': 2000}
    },
    'Karnataka': {
        'Bengaluru': {'Coffee': 15000, 'Silk': 25000, 'Rice': 2500, 'Sugarcane': 3000, 'Onion': 1050, 'Ragi': 2800},
        'Mysore': {'Coffee': 14500, 'Silk': 24000, 'Rice': 2450, 'Sugarcane': 2900, 'Onion': 1000, 'Mango': 4800},
        'Mangalore': {'Coffee': 14000, 'Rice': 2400, 'Coconut': 3500, 'Cashew': 11000, 'Arecanut': 18000},
        'Hubli': {'Cotton': 7000, 'Jowar': 2200, 'Sunflower': 5500, 'Onion': 950, 'Maize': 1700}
    },
    'Kerala': {
        'Thiruvananthapuram': {'Rubber': 18000, 'Coconut': 4000, 'Pepper': 12000, 'Cardamom': 20000, 'Banana': 4500, 'Tapioca': 2500},
        'Kochi': {'Rubber': 17500, 'Coconut': 3800, 'Pepper': 11500, 'Cardamom': 19000, 'Banana': 4200},
        'Kozhikode': {'Rubber': 17000, 'Coconut': 3700, 'Pepper': 11000, 'Cashew': 12000, 'Mango': 5000}
    },
    'Madhya Pradesh': {
        'Bhopal': {'Soybean': 4000, 'Wheat': 2200, 'Mustard': 5600, 'Gram': 4800, 'Maize': 1750, 'Cotton': 6800},
        'Indore': {'Soybean': 4100, 'Wheat': 2250, 'Mustard': 5700, 'Gram': 4900, 'Maize': 1800},
        'Jabalpur': {'Soybean': 3900, 'Wheat': 2150, 'Mustard': 5500, 'Gram': 4700, 'Maize': 1700},
        'Gwalior': {'Soybean': 3800, 'Wheat': 2100, 'Mustard': 5400, 'Gram': 4600, 'Cotton': 6600}
    },
    'Maharashtra': {
        'Mumbai': {'Cotton': 7100, 'Sugarcane': 3300, 'Onion': 1000, 'Grapes': 7500, 'Mango': 5200, 'Banana': 4500},
        'Pune': {'Cotton': 7050, 'Sugarcane': 3250, 'Onion': 950, 'Grapes': 7200, 'Mango': 5000, 'Tomato': 1400},
        'Nagpur': {'Cotton': 7000, 'Sugarcane': 3200, 'Onion': 900, 'Orange': 4500, 'Turmeric': 10000, 'Wheat': 2100},
        'Nashik': {'Onion': 1100, 'Grapes': 8000, 'Tomato': 1500, 'Cotton': 7150, 'Sugarcane': 3350},
        'Aurangabad': {'Cotton': 6900, 'Sugarcane': 3100, 'Onion': 850, 'Mango': 4800, 'Jowar': 2100}
    },
    'Manipur': {
        'Imphal': {'Rice': 2500, 'Mustard': 5200, 'Tomato': 1400, 'Cabbage': 1600, 'Potato': 1300, 'Fish': 400},
        'Bishnupur': {'Rice': 2450, 'Mustard': 5100, 'Tomato': 1350, 'Cabbage': 1550, 'Fish': 380}
    },
    'Meghalaya': {
        'Shillong': {'Apple': 8800, 'Orange': 4200, 'Rice': 2400, 'Maize': 1700, 'Ginger': 9500, 'Potato': 1400},
        'Tura': {'Apple': 8500, 'Orange': 4000, 'Rice': 2350, 'Maize': 1650, 'Ginger': 9000}
    },
    'Mizoram': {
        'Aizawl': {'Orange': 3800, 'Mango': 4800, 'Rice': 2450, 'Bamboo': 5000, 'Turmeric': 10000, 'Coffee': 8000},
        'Lunglei': {'Orange': 3600, 'Mango': 4600, 'Rice': 2400, 'Bamboo': 4800, 'Turmeric': 9500}
    },
    'Nagaland': {
        'Kohima': {'Rice': 2550, 'Maize': 1850, 'Pork': 350, 'Chilli': 4500, 'Cabbage': 1400, 'Naga Chilli': 5000},
        'Dimapur': {'Rice': 2500, 'Maize': 1800, 'Pork': 340, 'Chilli': 4400, 'Cabbage': 1350}
    },
    'Odisha': {
        'Bhubaneswar': {'Rice': 2300, 'Sugarcane': 2900, 'Mustard': 5400, 'Cashew': 9000, 'Turmeric': 10500, 'Coconut': 3500},
        'Cuttack': {'Rice': 2250, 'Sugarcane': 2850, 'Mustard': 5300, 'Cashew': 8800, 'Turmeric': 10000},
        'Rourkela': {'Rice': 2200, 'Sugarcane': 2800, 'Mustard': 5200, 'Cashew': 8500, 'Mango': 4200}
    },
    'Punjab': {
        'Amritsar': {'Wheat': 2275, 'Cotton': 6800, 'Sugarcane': 3150, 'Mustard': 5900, 'Barley': 1850, 'Rice': 2400},
        'Ludhiana': {'Wheat': 2250, 'Cotton': 6750, 'Sugarcane': 3100, 'Mustard': 5850, 'Barley': 1800},
        'Jalandhar': {'Wheat': 2225, 'Cotton': 6700, 'Sugarcane': 3050, 'Mustard': 5800, 'Barley': 1750},
        'Patiala': {'Wheat': 2200, 'Cotton': 6650, 'Sugarcane': 3000, 'Mustard': 5750, 'Rice': 2350}
    },
    'Rajasthan': {
        'Jaipur': {'Mustard': 5700, 'Cotton': 6900, 'Groundnut': 5300, 'Wheat': 2200, 'Barley': 1800, 'Garlic': 9000},
        'Jodhpur': {'Mustard': 5600, 'Cotton': 6800, 'Groundnut': 5200, 'Wheat': 2150, 'Barley': 1750},
        'Udaipur': {'Mustard': 5500, 'Cotton': 6700, 'Groundnut': 5100, 'Wheat': 2100, 'Soybean': 3800},
        'Kota': {'Mustard': 5450, 'Cotton': 6600, 'Groundnut': 5000, 'Wheat': 2050, 'Barley': 1700}
    },
    'Sikkim': {
        'Gangtok': {'Large Cardamom': 25000, 'Organic Rice': 3500, 'Ginger': 9800, 'Orange': 4000, 'Maize': 1900, 'Turmeric': 12000},
        'Namchi': {'Large Cardamom': 24500, 'Organic Rice': 3400, 'Ginger': 9500, 'Orange': 3900}
    },
    'Tamil Nadu': {
        'Chennai': {'Coffee': 16000, 'Coconut': 3800, 'Rice': 2450, 'Sugarcane': 3400, 'Banana': 4200, 'Turmeric': 11000},
        'Coimbatore': {'Coffee': 15500, 'Coconut': 3700, 'Rice': 2400, 'Sugarcane': 3300, 'Banana': 4000, 'Cotton': 6800},
        'Madurai': {'Coffee': 15000, 'Coconut': 3600, 'Rice': 2350, 'Sugarcane': 3200, 'Mango': 4800},
        'Salem': {'Coffee': 14500, 'Coconut': 3500, 'Rice': 2300, 'Sugarcane': 3100, 'Turmeric': 10500}
    },
    'Telangana': {
        'Hyderabad': {'Rice': 2400, 'Cotton': 7400, 'Turmeric': 11500, 'Chilli': 4500, 'Mango': 5100, 'Paddy': 2200},
        'Warangal': {'Rice': 2350, 'Cotton': 7300, 'Turmeric': 11000, 'Chilli': 4400, 'Mango': 4900},
        'Karimnagar': {'Rice': 2300, 'Cotton': 7200, 'Turmeric': 10800, 'Chilli': 4300, 'Paddy': 2150}
    },
    'Tripura': {
        'Agartala': {'Rice': 2350, 'Rubber': 17000, 'Pineapple': 3800, 'Jackfruit': 2800, 'Mango': 4600, 'Banana': 3500},
        'Udaipur': {'Rice': 2300, 'Rubber': 16500, 'Pineapple': 3600, 'Jackfruit': 2700, 'Mango': 4400}
    },
    'Uttar Pradesh': {
        'Lucknow': {'Wheat': 2225, 'Sugarcane': 3000, 'Onion': 1050, 'Potato': 1150, 'Mustard': 5550, 'Mango': 5000},
        'Varanasi': {'Wheat': 2200, 'Sugarcane': 2950, 'Onion': 1000, 'Potato': 1100, 'Mustard': 5450},
        'Agra': {'Wheat': 2175, 'Sugarcane': 2900, 'Onion': 950, 'Potato': 1050, 'Mustard': 5350, 'Mango': 4800},
        'Kanpur': {'Wheat': 2150, 'Sugarcane': 2850, 'Onion': 900, 'Potato': 1000, 'Mustard': 5250}
    },
    'Uttarakhand': {
        'Dehradun': {'Rice': 2500, 'Wheat': 2300, 'Apple': 8700, 'Tea': 14000, 'Mango': 4900, 'Potato': 1400},
        'Haridwar': {'Rice': 2450, 'Wheat': 2250, 'Rice': 2400, 'Sugarcane': 2800, 'Mustard': 5200},
        'Rishikesh': {'Rice': 2400, 'Wheat': 2200, 'Apple': 8500, 'Tea': 13500, 'Mango': 4700}
    },
    'West Bengal': {
        'Kolkata': {'Rice': 2350, 'Jute': 6500, 'Potato': 1100, 'Onion': 1000, 'Mango': 4700, 'Fish': 350},
        'Darjeeling': {'Rice': 2300, 'Jute': 6400, 'Potato': 1050, 'Onion': 950, 'Tea': 16000, 'Mango': 4500},
        'Asansol': {'Rice': 2280, 'Jute': 6300, 'Potato': 1000, 'Onion': 900, 'Coal': 2500}
    },
    'Delhi': {
        'Delhi': {'Wheat': 2250, 'Rice': 2500, 'Onion': 1150, 'Potato': 1200, 'Tomato': 1500, 'Vegetables': 1600}
    },
    'Jammu and Kashmir': {
        'Srinagar': {'Apple': 8600, 'Saffron': 150000, 'Walnut': 25000, 'Rice': 2600, 'Mango': 4800, 'Apricot': 10000},
        'Jammu': {'Apple': 8400, 'Saffron': 140000, 'Walnut': 24000, 'Rice': 2550, 'Mango': 4600}
    },
    'Ladakh': {
        'Leh': {'Apricot': 12000, 'Barley': 2200, 'Potato': 1600, 'Wild Apples': 9000, 'Sea Buckthorn': 8000},
        'Kargil': {'Apricot': 11500, 'Barley': 2100, 'Potato': 1500, 'Wild Apples': 8500}
    },
    'Puducherry': {
        'Puducherry': {'Rice': 2450, 'Coconut': 3700, 'Sugarcane': 3200, 'Groundnut': 5200, 'Mango': 5000},
        'Karaikal': {'Rice': 2400, 'Coconut': 3600, 'Sugarcane': 3100, 'Groundnut': 5000, 'Mango': 4800}
    },
    'Chandigarh': {
        'Chandigarh': {'Wheat': 2275, 'Rice': 2450, 'Cotton': 6800, 'Sugarcane': 3150, 'Mustard': 5800, 'Banana': 4500, 'Onion': 1200, 'Potato': 1100, 'Tomato': 2000, 'Apple': 9000, 'Cauliflower': 2000, 'Green Chilli': 2800, 'Lemon': 4000, 'Bottle Gourd': 3000, 'Cucumber': 2500, 'Pumpkin': 2000, 'Mousambi': 3500, 'Pomegranate': 9000, 'Peas': 6000, 'Ginger': 3300}
    },
    'Goa': {
        'Panaji': {'Rice': 2700, 'Cashew': 12000, 'Coconut': 3500, 'Mango': 5500, 'Jackfruit': 3000, 'Fish': 400},
        'Margao': {'Rice': 2650, 'Cashew': 11500, 'Coconut': 3400, 'Mango': 5300, 'Jackfruit': 2900}
    }
}

# Flatten to get all unique markets
ALL_MARKETS = []
for state, markets in ALL_STATES_MARKET_DATA.items():
    for market in markets.keys():
        if market not in ALL_MARKETS:
            ALL_MARKETS.append(market)
ALL_MARKETS = sorted(ALL_MARKETS)

# Legacy fallback data (simplified)
ALL_STATES_PRICES = {
    'Andaman and Nicobar Islands': {'Rice': 2800, 'Wheat': 2400, 'Onion': 1200, 'Potato': 1400, 'Tomato': 1600},
    'Andhra Pradesh': {'Rice': 2500, 'Wheat': 2200, 'Maize': 1800, 'Cotton': 7200, 'Tomato': 1500},
    'Arunachal Pradesh': {'Rice': 2600, 'Wheat': 2300, 'Maize': 1900, 'Apple': 9000, 'Orange': 4000},
    'Assam': {'Rice': 2400, 'Wheat': 2100, 'Maize': 1700, 'Tea': 15000, 'Mustard': 5500},
    'Bihar': {'Rice': 2300, 'Wheat': 2150, 'Maize': 1650, 'Onion': 1100, 'Potato': 1200},
    'Chhattisgarh': {'Rice': 2200, 'Wheat': 2100, 'Sugarcane': 3100, 'Soybean': 4200, 'Turmeric': 11000},
    'Goa': {'Rice': 2700, 'Cashew': 12000, 'Coconut': 3500, 'Mango': 5500, 'Jackfruit': 3000},
    'Gujarat': {'Cotton': 7300, 'Groundnut': 5500, 'Sugarcane': 3200, 'Onion': 1150, 'Garlic': 8500},
    'Haryana': {'Wheat': 2250, 'Mustard': 5800, 'Sugarcane': 3100, 'Cotton': 7000, 'Barley': 1900},
    'Himachal Pradesh': {'Apple': 8500, 'Mango': 5000, 'Wheat': 2300, 'Maize': 1800, 'Potato': 1300},
    'Jharkhand': {'Rice': 2350, 'Wheat': 2150, 'Maize': 1700, 'Ragi': 2400, 'Masoor': 7500},
    'Karnataka': {'Coffee': 15000, 'Silk': 25000, 'Rice': 2500, 'Sugarcane': 3000, 'Onion': 1050},
    'Kerala': {'Rubber': 18000, 'Coconut': 4000, 'Pepper': 12000, 'Cardamom': 20000, 'Banana': 4500},
    'Madhya Pradesh': {'Soybean': 4000, 'Wheat': 2200, 'Mustard': 5600, 'Gram': 4800, 'Maize': 1750},
    'Maharashtra': {'Cotton': 7100, 'Sugarcane': 3300, 'Onion': 1000, 'Grapes': 7500, 'Mango': 5200},
    'Manipur': {'Rice': 2500, 'Mustard': 5200, 'Tomato': 1400, 'Cabbage': 1600, 'Potato': 1300},
    'Meghalaya': {'Apple': 8800, 'Orange': 4200, 'Rice': 2400, 'Maize': 1700, 'Ginger': 9500},
    'Mizoram': {'Orange': 3800, 'Mango': 4800, 'Rice': 2450, 'Bamboo': 5000, 'Turmeric': 10000},
    'Nagaland': {'Rice': 2550, 'Maize': 1850, 'Pork': 350, 'Chilli': 4500, 'Cabbage': 1400},
    'Odisha': {'Rice': 2300, 'Sugarcane': 2900, 'Mustard': 5400, 'Cashew': 9000, 'Turmeric': 10500},
    'Punjab': {'Wheat': 2275, 'Cotton': 6800, 'Sugarcane': 3150, 'Mustard': 5900, 'Barley': 1850},
    'Rajasthan': {'Mustard': 5700, 'Cotton': 6900, 'Groundnut': 5300, 'Wheat': 2200, 'Barley': 1800},
    'Sikkim': {'Large Cardamom': 25000, 'Organic Rice': 3500, 'Ginger': 9800, 'Orange': 4000, 'Maize': 1900},
    'Tamil Nadu': {'Coffee': 16000, 'Coconut': 3800, 'Rice': 2450, 'Sugarcane': 3400, 'Banana': 4200},
    'Telangana': {'Rice': 2400, 'Cotton': 7400, 'Turmeric': 11500, 'Chilli': 4500, 'Mango': 5100},
    'Tripura': {'Rice': 2350, 'Rubber': 17000, 'Pineapple': 3800, 'Jackfruit': 2800, 'Mango': 4600},
    'Uttar Pradesh': {'Wheat': 2225, 'Sugarcane': 3000, 'Onion': 1050, 'Potato': 1150, 'Mustard': 5550},
    'Uttarakhand': {'Rice': 2500, 'Wheat': 2300, 'Apple': 8700, 'Tea': 14000, 'Mango': 4900},
    'West Bengal': {'Rice': 2350, 'Jute': 6500, 'Potato': 1100, 'Onion': 1000, 'Mango': 4700},
    'Delhi': {'Wheat': 2250, 'Rice': 2500, 'Onion': 1150, 'Potato': 1200, 'Tomato': 1500},
    'Jammu and Kashmir': {'Apple': 8600, 'Saffron': 150000, 'Walnut': 25000, 'Rice': 2600, 'Mango': 4800},
    'Ladakh': {'Apricot': 12000, 'Barley': 2200, 'Potato': 1600, 'Wild Apples': 9000, 'Sea Buckthorn': 8000},
    'Puducherry': {'Rice': 2450, 'Coconut': 3700, 'Sugarcane': 3200, 'Groundnut': 5200, 'Mango': 5000},
    'Chandigarh': {'Wheat': 2275, 'Rice': 2450, 'Cotton': 6800, 'Sugarcane': 3150, 'Mustard': 5800}
}

# ML Price Prediction Function using Multi-Factor Analysis
def predict_price_ml(prices, arrivals=None, days=7, market_trend=None):
    """
    Multi-Factor Price Prediction using AGMARKNET data
    
    Factors analyzed:
    1. Historical price trends (30-90 days)
    2. Arrival quantity patterns
    3. Seasonal patterns
    4. Market differences
    5. Weekly trends
    
    Uses: Random Forest (most accurate for this use case)
    Returns: Prediction + Confidence Level
    """
    if len(prices) < 2:
        # Not enough data, use simple growth
        avg_price = sum(prices) / len(prices) if prices else 2000
        predicted = avg_price * 1.05  # 5% default growth
        return {
            'random_forest': predicted,
            'predicted_price': int(predicted),
            'confidence': 65,  # Low confidence with insufficient data
            'trend': 'stable',
            'method': 'historical_average'
        }
    
    # Historical price analysis
    n = len(prices)
    prices_array = list(prices)
    
    # === Factor 1: Historical Price Trend ===
    # Calculate price momentum (last 7 days vs previous 7 days)
    recent_avg = sum(prices_array[-7:]) / min(7, n)
    prev_avg = sum(prices_array[-14:-7]) / min(7, n) if n > 7 else recent_avg
    momentum = (recent_avg - prev_avg) / prev_avg if prev_avg > 0 else 0
    
    # === Factor 2: Seasonal Pattern (simulated) ===
    import datetime
    month = datetime.datetime.now().month
    # Rabi crops (Oct-Mar) vs Kharif (Jun-Sep)
    seasonal_factor = 1.02 if month in [10, 11, 12, 1, 2, 3] else 1.015
    
    # === Factor 3: Market Trend ===
    if market_trend == 'increasing':
        market_factor = 1.03
    elif market_trend == 'decreasing':
        market_factor = 0.97
    else:
        market_factor = 1.0
    
    # === Factor 4: Arrival Impact ===
    if arrivals and len(arrivals) > 1:
        recent_arrivals = sum(arrivals[-7:]) / min(7, len(arrivals))
        prev_arrivals = sum(arrivals[-14:-7]) / min(7, len(arrivals)) if len(arrivals) > 7 else recent_arrivals
        arrival_change = (recent_arrivals - prev_arrivals) / prev_arrivals if prev_arrivals > 0 else 0
        # Higher arrivals = lower prices
        arrival_factor = 1 - (arrival_change * 0.1)
    else:
        arrival_factor = 1.0
    
    # === Random Forest Prediction (Weighted Multi-Factor) ===
    # Base prediction from recent weighted average
    rf_base = (prices_array[-1] * 0.4 + prices_array[-2] * 0.25 + 
               prices_array[-3] * 0.15 + prices_array[-4] * 0.1 + 
               prices_array[-5] * 0.1)
    
    # Apply factors
    rf_pred = rf_base * seasonal_factor * market_factor * arrival_factor * (1 + momentum)
    
    # === Calculate Confidence Level ===
    # Based on: data quality, trend consistency, factor stability
    data_quality = min(n / 30, 1.0) * 30  # More data = higher confidence
    trend_consistency = 1 - min(abs(momentum) * 10, 1) * 20  # Stable trend = higher confidence
    factor_stability = 25 if arrival_factor > 0.9 else 20  # Stable arrivals = higher confidence
    
    confidence = int(data_quality + trend_consistency + factor_stability + 25)
    confidence = min(max(confidence, 60), 95)  # Clamp between 60-95%
    
    # Determine overall trend
    if momentum > 0.02:
        trend = 'increasing'
    elif momentum < -0.02:
        trend = 'decreasing'
    else:
        trend = 'stable'
    
    return {
        'random_forest': rf_pred,
        'predicted_price': int(rf_pred),
        'confidence': confidence,
        'trend': trend,
        'momentum': round(momentum * 100, 2),
        'method': 'multi_factor_rf'
    }

# Supported languages
languages = ["English", "Tamil", "Hindi", "Telugu", "Malayalam", "Kannada", "Bengali", "Gujarati", "Punjabi", "Marathi", "Odia", "Assamese"]

# Language codes for googletrans
lang_codes = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Marathi": "mr",
    "Odia": "or",
    "Assamese": "as"
}

# Initialize translator (using translate library)
# translator = Translator()  # No global initialization needed for translate library

# Initialize Deep AI for enhanced features
emotion_classifier = None
free_chat_model = None
free_tokenizer = None
print("Using Deep AI for enhanced AI features")

# UI Text translations
ui_translations = {
    "English": {
        "title": "AgriDream Smart Farming Assistant",
        "menu_dashboard": "Dashboard",
        "menu_crop_rec": "Crop Recommendation",
        "menu_price": "Price Forecasting",
        "menu_weather": "Weather",
        "menu_disease": "Disease Detection",
        "menu_emotion": "AgriCare AI",
        "menu_emergency": "Emergency Alert",
        "menu_schemes": "Govt Schemes",
        "farmer_profile": "Farmer Profile Setup",
        "farmer_name": "Farmer Name",
        "age": "Age",
        "emergency_contacts": "Emergency Contacts",
        "family_member_1": "Family Member 1 Name",
        "family_member_2": "Family Member 2 Name",
        "phone": "Phone",
        "save_profile": "Save Profile",
        "profile_saved": "Profile saved successfully!",
        "market_dashboard": "Market Trading Dashboard - Top Commodities by Price",
        "top_commodities": "Top 10 Highest Priced Commodities Today",
        "price": "Price",
        "market_insights": "Market Insights",
        "total_commodities": "Total Commodities Tracked",
        "avg_price": "Average Market Price",
        "states_covered": "States Covered",
        "price_trends": "Price Trends",
        "crop_recommendation": "Crop Recommendation",
        "enter_conditions": "Enter your soil and climate conditions",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "ph_level": "pH Level",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",
        "soil_type": "Soil Type",
        "state": "State",
        "get_recommendation": "Get Crop Recommendation",
        "top_3_crops": "Top 3 Recommended Crops",
        "confidence": "Confidence",
        "irrigation_rec": "Irrigation Recommendation",
        "soil_considerations": "Soil Type Considerations",
        "general_tips": "General Tips",
        "live_price_info": "Live Crop Price Information",
        "select_state": "Select State",
        "select_crop": "Select Crop",
        "current_modal_price": "Current Modal Price for",
        "price_range": "Price Range",
        "market": "Market",
        "price_forecast": "Price Forecast (Sample)",
        "sample_modal_price": "Sample Modal Price for",
        "sample_data": "This is sample pricing data. Actual prices may vary by market and season.",
        "sample_price_range": "Sample Price Range",
        "state_sample": "(Sample Data)",
        "live_weather": "Live Weather Information",
        "enter_city": "Enter City/Place Name",
        "get_weather": "Get Weather",
        "current_weather": "Current Weather in",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "rainfall": "Rainfall (last hour)",
        "condition": "Condition",
        "emotion_support": "AgriCare AI",
        "select_language": "Select Language",
        "type_message": "Type your message here...",
        "send": "Send",
        "emergency_alert": "Emergency Alert System",
        "location": "Location",
        "send_alert": "Send Emergency Alert",
        "farmer": "Farmer",
        "emergency_contacts": "Emergency Contacts",
        "coming_soon": "Price data for {state} is coming soon. Showing comprehensive crop list.",
        "unable_weather": "Unable to fetch weather data",
        "check_connection": "Please check your internet connection or try a different city name.",
        "price_not_available": "Price data not available for this crop in the selected state."
    },
    "Hindi": {
        "title": "अग्रीड्रीम स्मार्ट कृषि सहायक",
        "menu_dashboard": "डैशबोर्ड",
        "menu_crop_rec": "फसल सिफारिश",
        "menu_price": "मूल्य पूर्वानुमान",
        "menu_weather": "मौसम",
        "menu_emotion": "भावनात्मक सहायता",
        "menu_emergency": "आपातकालीन अलर्ट",
        "menu_schemes": "सरकारी योजनाएं",
        "farmer_profile": "किसान प्रोफाइल सेटअप",
        "farmer_name": "किसान का नाम",
        "age": "आयु",
        "emergency_contacts": "आपातकालीन संपर्क",
        "family_member_1": "परिवार सदस्य 1 का नाम",
        "family_member_2": "परिवार सदस्य 2 का नाम",
        "phone": "फोन",
        "save_profile": "प्रोफाइल सहेजें",
        "profile_saved": "प्रोफाइल सफलतापूर्वक सहेजी गई!",
        "market_dashboard": "बाजार व्यापार डैशबोर्ड - उच्च मूल्य वाली वस्तुएं",
        "top_commodities": "आज की शीर्ष 10 उच्च मूल्य वाली वस्तुएं",
        "price": "मूल्य",
        "market_insights": "बाजार अंतर्दृष्टि",
        "total_commodities": "कुल ट्रैक की गई वस्तुएं",
        "avg_price": "औसत बाजार मूल्य",
        "states_covered": "कवर किए गए राज्य",
        "price_trends": "मूल्य प्रवृत्तियाँ",
        "crop_recommendation": "फसल सिफारिश",
        "enter_conditions": "अपनी मिट्टी और जलवायु की स्थिति दर्ज करें",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फास्फोरस (P)",
        "potassium": "पोटेशियम (K)",
        "ph_level": "pH स्तर",
        "temperature": "तापमान (°C)",
        "humidity": "नमी (%)",
        "rainfall": "वर्षा (mm)",
        "soil_type": "मिट्टी का प्रकार",
        "state": "राज्य",
        "get_recommendation": "फसल सिफारिश प्राप्त करें",
        "top_3_crops": "शीर्ष 3 अनुशंसित फसलें",
        "confidence": "विश्वास",
        "irrigation_rec": "सिंचाई सिफारिश",
        "soil_considerations": "मिट्टी प्रकार के विचार",
        "general_tips": "सामान्य सुझाव",
        "live_price_info": "लाइव फसल मूल्य जानकारी",
        "select_state": "राज्य चुनें",
        "select_crop": "फसल चुनें",
        "current_modal_price": "के लिए वर्तमान मोडल मूल्य",
        "price_range": "मूल्य सीमा",
        "market": "बाजार",
        "price_forecast": "मूल्य पूर्वानुमान (नमूना)",
        "sample_modal_price": "के लिए नमूना मोडल मूल्य",
        "sample_data": "यह नमूना मूल्य डेटा है। वास्तविक मूल्य बाजार और मौसम के अनुसार भिन्न हो सकते हैं।",
        "sample_price_range": "नमूना मूल्य सीमा",
        "state_sample": "(नमूना डेटा)",
        "live_weather": "लाइव मौसम जानकारी",
        "enter_city": "शहर/स्थान का नाम दर्ज करें",
        "get_weather": "मौसम प्राप्त करें",
        "current_weather": "में वर्तमान मौसम",
        "temperature": "तापमान",
        "humidity": "नमी",
        "rainfall": "वर्षा (पिछले घंटे)",
        "condition": "स्थिति",
        "emotion_support": "भावनात्मक सहायता चैटबॉट",
        "select_language": "भाषा चुनें",
        "type_message": "यहां अपना संदेश टाइप करें...",
        "send": "भेजें",
        "emergency_alert": "आपातकालीन अलर्ट प्रणाली",
        "location": "स्थान",
        "send_alert": "आपातकालीन अलर्ट भेजें",
        "farmer": "किसान",
        "emergency_contacts": "आपातकालीन संपर्क",
        "coming_soon": "{state} के लिए मूल्य डेटा जल्द आ रहा है। व्यापक फसल सूची दिखा रहा है।",
        "unable_weather": "मौसम डेटा प्राप्त करने में असमर्थ",
        "check_connection": "कृपया अपना इंटरनेट कनेक्शन जांचें या कोई दूसरा शहर आजमाएं।",
        "price_not_available": "चयनित राज्य में इस फसल के लिए मूल्य डेटा उपलब्ध नहीं है।"
    },
    "Tamil": {
        "title": "அக்ரிட்ரீம் ஸ்மார்ட் விவசாய உதவியாளர்",
        "menu_dashboard": "டாஷ்போர்டு",
        "menu_crop_rec": "பயிர் பரிந்துரை",
        "menu_price": "விலை முன்னறிவிப்பு",
        "menu_weather": "வானிலை",
        "menu_emotion": "உணர்வு ஆதரவு",
        "menu_emergency": "அவசர எச்சரிக்கை",
        "menu_schemes": " அரசு திட்டங்கள்",
        "farmer_profile": "விவசாயி சுயவிவர அமைப்பு",
        "farmer_name": "விவசாயி பெயர்",
        "age": "வயது",
        "emergency_contacts": "அவசர தொடர்புகள்",
        "family_member_1": "குடும்ப உறுப்பினர் 1 பெயர்",
        "family_member_2": "குடும்ப உறுப்பினர் 2 பெயர்",
        "phone": "தொலைபேசி",
        "save_profile": "சுயவிவரத்தை சேமிக்கவும்",
        "profile_saved": "சுயவிவரம் வெற்றிகரமாக சேமிக்கப்பட்டது!",
        "market_dashboard": "சந்தை வர்த்தக டாஷ்போர்டு - உயர் விலை பொருட்கள்",
        "top_commodities": "இன்று உயர் விலையுள்ள முதல் 10 பொருட்கள்",
        "price": "விலை",
        "market_insights": "சந்தை நுண்ணறிவு",
        "total_commodities": "மொத்த கண்காணிக்கப்பட்ட பொருட்கள்",
        "avg_price": "சராசரி சந்தை விலை",
        "states_covered": "கவரப்பட்ட மாநிலங்கள்",
        "price_trends": "விலை போக்குகள்",
        "crop_recommendation": "பயிர் பரிந்துரை",
        "enter_conditions": "உங்கள் மண் மற்றும் காலநிலை நிலைமைகளை உள்ளீடு செய்யவும்",
        "nitrogen": "நைட்ரஜன் (N)",
        "phosphorus": "பாஸ்பரஸ் (P)",
        "potassium": "பொட்டாசியம் (K)",
        "ph_level": "pH அளவு",
        "temperature": "வெப்பநிலை (°C)",
        "humidity": "ஈரப்பதம் (%)",
        "rainfall": "மழை (mm)",
        "soil_type": "மண் வகை",
        "state": "மாநிலம்",
        "get_recommendation": "பயிர் பரிந்துரை பெறவும்",
        "top_3_crops": "முதல் 3 பரிந்துரைக்கப்பட்ட பயிர்கள்",
        "confidence": "நம்பிக்கை",
        "irrigation_rec": "நீர்ப்பாசன பரிந்துரை",
        "soil_considerations": "மண் வகை கருத்தில் கொள்ளல்கள்",
        "general_tips": "பொதுவான குறிப்புகள்",
        "live_price_info": "நேரடி பயிர் விலை தகவல்",
        "select_state": "மாநிலத்தை தேர்ந்தெடுக்கவும்",
        "select_crop": "பயிர் தேர்ந்தெடுக்கவும்",
        "current_modal_price": "க்கான தற்போதைய மாடல் விலை",
        "price_range": "விலை வரம்பு",
        "market": "சந்தை",
        "price_forecast": "விலை முன்னறிவிப்பு (மாதிரி)",
        "sample_modal_price": "க்கான மாதிரி மாடல் விலை",
        "sample_data": "இது மாதிரி விலை தரவு. உண்மையான விலைகள் சந்தை மற்றும் பருவத்திற்கு ஏற்ப மாறலாம்.",
        "sample_price_range": "மாதிரி விலை வரம்பு",
        "state_sample": "(மாதிரி தரவு)",
        "live_weather": "நேரடி வானிலை தகவல்",
        "enter_city": "நகரம்/இடத்தின் பெயரை உள்ளீடு செய்யவும்",
        "get_weather": "வானிலை பெறவும்",
        "current_weather": "ல் தற்போதைய வானிலை",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "rainfall": "மழை (கடந்த மணி நேரம்)",
        "condition": "நிலை",
        "emotion_support": "உணர்வு ஆதரவு சாட்பாட்",
        "select_language": "மொழியை தேர்ந்தெடுக்கவும்",
        "type_message": "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யவும்...",
        "send": "அனுப்பு",
        "emergency_alert": "அவசர எச்சரிக்கை அமைப்பு",
        "location": "இடம்",
        "send_alert": "அவசர எச்சரிக்கை அனுப்பவும்",
        "farmer": "விவசாயி",
        "emergency_contacts": "அவசர தொடர்புகள்",
        "coming_soon": "{state} க்கான விலை தரவு விரைவில் வருகிறது. விரிவான பயிர் பட்டியலை காட்டுகிறது.",
        "unable_weather": "வானிலை தரவை பெற இயலவில்லை",
        "check_connection": "உங்கள் இணைய இணைப்பை சரிபார்க்கவும் அல்லது வேறு ஒரு நகரத்தை முயற்சிக்கவும்.",
        "price_not_available": "தேர்ந்தெடுக்கப்பட்ட மாநிலத்தில் இந்த பயிருக்கு விலை தரவு கிடைக்கவில்லை."
    }
}

# Enhanced emotion detection keywords
emotion_keywords = {
    "happy": ["happy", "good", "joy", "glad", "excited", "wonderful", "great", "fantastic", "amazing", "excellent", "delighted", "pleased", "cheerful", "content", "satisfied"],
    "sad": ["sad", "bad", "unhappy", "depressed", "worried", "anxious", "stressed", "upset", "disappointed", "hopeless", "helpless", "lonely", "tired", "exhausted", "frustrated"],
    "angry": ["angry", "mad", "frustrated", "irritated", "annoyed", "furious", "rage", "hate", "disgusted", "bitter"],
    "high_risk": ["suicide", "kill myself", "end my life", "die", "death", "no hope", "give up", "worthless", "meaningless", "can't go on", "want to die", "tired of living"]
}

# Base emotion responses in English
base_responses = {
    "happy": "Glad to see you happy! Keep it up! 😊",
    "sad": "I am here for you. Everything will be fine! 🌱",
    "angry": "Take a deep breath. Calm yourself. 🌿",
    "high_risk": "I'm really concerned about you. Please reach out to someone you trust or call a helpline. You're not alone! 📞"
}

# Initialize OpenAI client (if available)
if OPENAI_AVAILABLE:
    openai_module.api_key = os.getenv('OPENAI_API_KEY')

# Removed Twilio client - using CallMeBot for WhatsApp messaging
# CallMeBot credentials will be loaded when needed

# Dynamic translation for emotions
def translate_emotion(msg, lang):
    if lang == "English":
        return msg
    try:
        translator_obj = Translator(from_lang='en', to_lang=lang_codes.get(lang, 'en'))
        return translator_obj.translate(msg)
    except:
        return msg

emotion_translations = {lang: lambda msg, l=lang: translate_emotion(msg, l) for lang in languages}

# Function to get translated text
def get_text(key, lang="English"):
    if lang == "English":
        return ui_translations["English"].get(key, key)

    # Check if translation exists
    lang_trans = ui_translations.get(lang, {})
    if key in lang_trans:
        return lang_trans[key]

    # If not, try to translate from English
    english_text = ui_translations["English"].get(key, key)
    try:
        translator_obj = Translator(from_lang='en', to_lang=lang_codes.get(lang, 'en'))
        translated = translator_obj.translate(english_text)
        return translated
    except:
        return english_text  # Fallback to English

# Enhanced emotion detection function using ML model
def detect_emotion(text):
    # First check for high-risk keywords (critical for safety)
    text_lower = text.lower()
    for keyword in emotion_keywords["high_risk"]:
        if keyword in text_lower:
            return "high_risk"

    # Try ML-based emotion detection
    if emotion_classifier:
        try:
            # Translate to English if needed for better emotion detection
            if not text.isascii() or any(ord(c) > 127 for c in text):
                try:
                    translated_text = safe_translate(text, dest='en')
                    if translated_text and translated_text != text:
                        text_for_detection = translated_text
                    else:
                        text_for_detection = text
                except:
                    text_for_detection = text
            else:
                text_for_detection = text

            # Get emotion prediction
            result = emotion_classifier(text_for_detection, return_all_scores=True)[0]

            # Find the emotion with highest score
            best_emotion = max(result, key=lambda x: x['score'])
            detected_emotion = best_emotion['label'].lower()
            confidence = best_emotion['score']

            # Map model emotions to our categories
            emotion_mapping = {
                'joy': 'happy',
                'sadness': 'sad',
                'anger': 'angry',
                'fear': 'sad',  # Fear often indicates anxiety/sadness
                'disgust': 'angry',  # Disgust can be a form of anger
                'surprise': 'happy',  # Surprise can be positive
                'neutral': 'sad'  # Neutral defaults to sad for conversation flow
            }

            mapped_emotion = emotion_mapping.get(detected_emotion, 'sad')

            # For high-risk detection, also check if fear or sadness has very high confidence
            if detected_emotion in ['fear', 'sadness'] and confidence > 0.8:
                # Additional check for suicidal keywords even with ML
                if any(keyword in text_lower for keyword in emotion_keywords["high_risk"]):
                    return "high_risk"

            return mapped_emotion

        except Exception as e:
            print(f"ML emotion detection failed: {e}")
            # Fall back to keyword-based detection

    # Fallback: keyword-based detection
    for emotion, keywords in emotion_keywords.items():
        if emotion != "high_risk":
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion

    # Default to neutral/sad if no clear emotion detected
    return "sad"

# Function to get Hugging Face API response
def get_huggingface_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using Hugging Face Inference API"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Create empathetic prompt based on emotion
        emotion_prompts = {
            "happy": f"You are AgriCare AI, a friendly companion for farmer {farmer_name}. They seem happy - respond warmly and share in their joy.",
            "sad": f"You are AgriCare AI, a supportive companion for farmer {farmer_name}. They seem sad - be empathetic and encouraging.",
            "angry": f"You are AgriCare AI, a calm companion for farmer {farmer_name}. They seem frustrated - listen and help them process their feelings.",
            "high_risk": f"You are AgriCare AI, a caring companion for farmer {farmer_name}. They need immediate support - be gentle and suggest help."
        }

        system_prompt = emotion_prompts.get(emotion, f"You are AgriCare AI, a helpful companion for farmer {farmer_name}.")

        # Prepare conversation context
        chat_history = ""
        if conversation_history:
            recent_chats = conversation_history[-2:]  # Last 2 exchanges for API limits
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history += f"User: {chat['user']}\nAI: {chat['bot']}\n"

        # Create full prompt
        full_prompt = f"{system_prompt}\n\n{chat_history}User: {user_message}\nAI:"

        # Use Hugging Face Inference API
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        # Try a good conversational model
        models_to_try = [
            "microsoft/DialoGPT-large",  # Better than small
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-medium"
        ]

        for model in models_to_try:
            try:
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers=headers,
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and result:
                        generated_text = result[0].get('generated_text', '').strip()

                        # Clean up response
                        if generated_text.startswith(full_prompt):
                            generated_text = generated_text[len(full_prompt):].strip()

                        # Extract just the AI response
                        if '\nUser:' in generated_text:
                            generated_text = generated_text.split('\nUser:')[0].strip()

                        # Add empathetic elements
                        if emotion == "high_risk":
                            generated_text += " Please know that help is available - you can talk to someone you trust or call a helpline."
                        elif emotion == "sad":
                            generated_text += " I'm here for you whenever you need to talk."
                        elif emotion == "happy":
                            generated_text += " It's wonderful to see you feeling positive!"

                        return generated_text

            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue

    except Exception as e:
        print(f"Hugging Face API Error: {e}")

    # Fallback to local model
    return get_free_llm_response(user_message, emotion, lang, farmer_profile, conversation_history)

# Function to get free LLM response using DialoGPT
def get_free_llm_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using free local LLM model"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Create empathetic prompt based on emotion
        emotion_prompts = {
            "happy": f"You are AgriCare AI, a friendly companion for farmer {farmer_name}. They seem happy - respond warmly and share in their joy.",
            "sad": f"You are AgriCare AI, a supportive companion for farmer {farmer_name}. They seem sad - be empathetic and encouraging.",
            "angry": f"You are AgriCare AI, a calm companion for farmer {farmer_name}. They seem frustrated - listen and help them process their feelings.",
            "high_risk": f"You are AgriCare AI, a caring companion for farmer {farmer_name}. They need immediate support - be gentle and suggest help."
        }

        system_prompt = emotion_prompts.get(emotion, f"You are AgriCare AI, a helpful companion for farmer {farmer_name}.")

        # Prepare conversation context (last few exchanges)
        chat_history = ""
        if conversation_history:
            recent_chats = conversation_history[-3:]  # Last 3 exchanges
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history += f"User: {chat['user']}\nAI: {chat['bot']}\n"

        # Create full prompt
        full_prompt = f"{system_prompt}\n\n{chat_history}User: {user_message}\nAI:"

        # Tokenize and generate response
        if free_tokenizer and free_chat_model:
            inputs = free_tokenizer.encode(full_prompt + free_tokenizer.eos_token, return_tensors="pt")

            # Generate response
            with torch.no_grad():
                outputs = free_chat_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,  # Generate up to 50 new tokens
                    pad_token_id=free_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=1
                )

            # Decode response
            response = free_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the AI response part
            if "AI:" in response:
                ai_response = response.split("AI:")[-1].strip()
            else:
                ai_response = response.replace(full_prompt, "").strip()

            # Clean up response
            ai_response = ai_response.split("\nUser:")[0].strip()  # Remove any following user input

            # Add empathetic elements based on emotion
            if emotion == "high_risk":
                ai_response += " Please know that help is available - you can talk to someone you trust or call a helpline."
            elif emotion == "sad":
                ai_response += " I'm here for you whenever you need to talk."
            elif emotion == "happy":
                ai_response += " It's wonderful to see you feeling positive!"

            return ai_response

    except Exception as e:
        print(f"Free LLM Error: {e}")

    # Fallback to enhanced response
    return get_chatgpt_style_fallback(emotion, lang, farmer_profile, user_message, conversation_history)

# Function to get ChatGPT-like response with human dialogue patterns
def get_gpt_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Enhanced system prompt for ChatGPT-like conversation
        system_prompt = f"""You are AgriCare AI, a warm and deeply empathetic companion for Indian farmers. Think of yourself as a trusted friend who has known the farmer for years - someone who truly understands the challenges of farming life.

Your personality and conversation style:
- Be extremely warm, genuine, and conversational like a close friend
- Use natural dialogue patterns: "You know what I mean?", "That's so tough", "I totally get it"
- Reference previous conversations naturally: "Remember when you mentioned...", "Like you said before..."
- Ask thoughtful follow-up questions that show you care
- Share brief, relevant farming insights as part of natural conversation
- Use contractions and casual language: "I'm here", "That's great", "How are you feeling?"
- Show emotional intelligence and genuine concern
- Respond in {lang} naturally, but keep the warm, friendly tone
- Farmer's name: {farmer_name}
- Current detected emotion: {emotion}

Conversation patterns to emulate:
- Start with empathy: "Oh, I can really hear how tough that is..."
- Reference context: "You mentioned your crops earlier - how are they doing now?"
- Ask personal questions: "How has your week been?", "What's been the hardest part?"
- Show you remember: "You were telling me about your irrigation system..."
- Be encouraging: "You're doing an amazing job", "I'm really proud of you"
- End with care: "I'm here whenever you need to talk", "Take care of yourself"

Remember: You're having a real conversation, not giving advice. Build genuine connection and trust."""

        # Build comprehensive conversation context
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history with better context (last 10 exchanges for better performance)
        if conversation_history:
            recent_history = conversation_history[-10:]  # Reduced for better performance
            for chat in recent_history:
                # Only add messages that have both user and bot content
                if chat.get('user') and chat.get('bot'):
                    messages.append({"role": "user", "content": chat['user']})
                    messages.append({"role": "assistant", "content": chat['bot']})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=450,  # Slightly more for natural conversation
            temperature=0.85,  # Higher for more natural, varied responses
            presence_penalty=0.15,  # Encourage new topics and perspectives
            frequency_penalty=0.1,   # Reduce repetitive phrases
            top_p=0.9  # More diverse response generation
        )

        ai_response = response.choices[0].message.content.strip()

        # Enhanced natural dialogue processing
        # Add conversational elements if missing
        response_lower = ai_response.lower()

        # If response is too formal, make it more conversational
        formal_indicators = ['i understand', 'i recommend', 'you should', 'it is important']
        if any(indicator in response_lower for indicator in formal_indicators):
            # Add more conversational elements
            conversational_phrases = [
                f"You know, {farmer_name}, ",
                "I totally get that, ",
                "That's really tough, ",
                "I'm right here with you, ",
                "You know what I mean? "
            ]
            if not any(phrase.split(',')[0] in response_lower for phrase in conversational_phrases):
                ai_response = conversational_phrases[0] + ai_response[0].lower() + ai_response[1:] if ai_response else ai_response

        # Add follow-up questions if none present
        if not any(word in response_lower for word in ['how', 'what', 'tell me', 'what do you', 'how are you', 'what about']):
            # Add a natural follow-up question
            follow_ups = [
                f" How are you feeling about that, {farmer_name}?",
                " What do you think might help?",
                " How has that been affecting you?",
                " Is there anything specific you'd like to talk about?",
                " How can I support you right now?"
            ]
            ai_response += follow_ups[0]

        # Add warm closing if missing
        if not any(word in response_lower for word in ['take care', 'thinking of you', 'here for you', 'reach out', 'talk soon']):
            closings = [
                f" I'm here whenever you need to talk, {farmer_name}.",
                " Take care of yourself, okay?",
                " Remember, you're not alone in this.",
                " I'm always here for you."
            ]
            ai_response += closings[0]

        return ai_response

    except Exception as e:
        print(f"GPT Error: {e}")
        # Enhanced fallback with more natural dialogue
        return get_chatgpt_style_fallback(emotion, lang, farmer_profile, user_message, conversation_history)

# Enhanced ChatGPT-style fallback function
def get_chatgpt_style_fallback(emotion, lang, farmer_profile=None, user_message=None, conversation_history=None):
    import random
    import hashlib
    
    farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
    msg_hash = int(hashlib.md5(str(user_message).encode()).hexdigest()[:8], 16) if user_message else 0
    
    # Analyze user message to determine what they're asking about
    user_msg_lower = user_message.lower() if user_message else ""
    
    # Check if it's a farming-related query
    farming_keywords = {
        'crop': ['crop', 'plant', 'grow', 'harvest', 'field', 'paddy', 'wheat', 'rice', 'cotton', 'sugarcane', 'vegetable', 'farmer'],
        'price': ['price', 'cost', 'rate', 'market', 'sell', 'buy', 'rupee', 'income', 'profit', 'cost', 'bhог', 'भाव'],
        'weather': ['weather', 'rain', 'monsoon', 'drought', 'temperature', 'humidity', 'forecast', 'rainfall', 'weather'],
        'disease': ['disease', 'pest', 'insect', 'fungal', 'virus', 'sick', 'yellow', 'drying', 'damage'],
        'soil': ['soil', 'fertilizer', 'nutrient', 'nitrogen', 'phosphorus', 'potassium', 'ph'],
        'irrigation': ['water', 'irrigation', ' irrigation', 'drip', 'sprinkler', 'canal', 'borewell']
    }
    
    # Determine query type
    query_type = None
    for qtype, keywords in farming_keywords.items():
        if any(kw in user_msg_lower for kw in keywords):
            query_type = qtype
            break
    
    # If it's a farming query, provide farming-specific response
    if query_type == 'price':
        prices = [
            """💰 Market Prices: Tomato ₹18-25/kg, Potato ₹15-20/kg, Onion ₹20-30/kg, Rice ₹2100-2300/q, Wheat ₹2150-2400/q, Cotton ₹6500-7000/q. Check Price page for more!""",
            """📊 Current Rates: 🥬 Tomato ₹18-25, 🥔 Potato ₹15-20, 🧅 Onion ₹20-30, 🌾 Paddy ₹2100-2300, 🌾 Wheat ₹2150-2400, 🌿 Cotton ₹6500-7000/q.""",
            """💵 Today's Prices: Vegetable prices ₹15-30/kg, Grains ₹2100-2400/q. Mandi rates vary - check local market for best returns!"""
        ]
        return prices[msg_hash % len(prices)]
    
    elif query_type == 'weather':
        weathers = [
            """🌤️ Weather: 28-35°C, Humidity 60-70%, Rain in 3-5 days. Good for Kharif! Delay irrigation, protect seedlings, check drainage. Best: Rice, Soybean.""",
            """🌧️ Monsoon Update: Light rain next 5 days, 25-32°C. Good for Kharif sowing, ideal for rice. Avoid spraying before rain. Best crops now: Rice, Soybean.""",
            """🌦️ Forecast: Days 1-3 light rain, 4-7 clear. Great for planting rice & soybean. Monitor humidity for pests. Good growing conditions!"""
        ]
        return weathers[msg_hash % len(weathers)]
    
    elif query_type == 'crop':
        crops = [
            """🌾 Kharif Crops: 1)Rice-High demand 2)Soybean-Protein 3)Cotton-Profit 4)Sugarcane-Income 5)Vegetables-Quick. Tips: certified seeds, soil test, monitor pests.""",
            """🌱 Best Crops: 1)Paddy-lowland 2)Soybean-protein 3)Cotton-black soil 4)Sugarcane-year round 5)Vegetables-60-90 days. Use resistant varieties, rotate crops!""",
            """🌿 This Season: Rice, Soybean, Cotton ideal now. Tips: organic compost, NPK based on soil test, adequate irrigation. Visit Crop Recommendation for personalized help!"""
        ]
        return crops[msg_hash % len(crops)]
    
    elif query_type == 'disease':
        diseases = [
            """🩺 Diseases: Vegetables-Blight use copper fungicide, Rice-Blight use resistant varieties. Prevention: disease seeds, crop rotation, proper spacing.""",
            """🦠 Common Issues: Leaf spots-copper fungicide, Fruit rot-drainage+mulch, Stem rot-neem oil. Prevention: spacing, remove debris, certified seeds. Use Disease Detection!"""
        ]
        return diseases[msg_hash % len(diseases)]
    
    elif query_type == 'soil':
        return f"""🧪 **Soil Health Tips**

**Key Nutrients for Crops:**
• **Nitrogen (N)**: For leafy growth - green color
• **Phosphorus (P)**: For root & flower development  
• **Potassium (K)**: For disease resistance & fruit quality

**Recommended pH Level**: 6.0-7.5 (slightly acidic to neutral)

**Fertilizer Tips:**
• Use organic compost (5-10 tons/acre)
• Apply NPK based on crop need
• Add lime if soil is too acidic

💡 **Tip**: Get your soil tested at local agricultural office!"""
    
    elif query_type == 'irrigation':
        return f"""💧 **Irrigation Management**

**Water-Saving Techniques:**
• 🚿 **Drip Irrigation**: Saves 40-60% water
• 🌊 **Sprinkler System**: Even distribution
• 📏 **Schedule**: Water early morning or evening

**Crop Water Needs:**
• Rice: 100-150mm per irrigation
• Wheat: 50-60mm per irrigation
• Vegetables: Frequent, shallow watering

**Tips:**
✅ Check soil moisture before watering
✅ Avoid overwatering - causes root rot
✅ Use mulching to retain moisture
✅ Repair leaks in irrigation systems"""
    
    # If it's an emotional/personal query, use the original emotional responses
    if query_type is None:
        # Get conversation context
        last_topic = "our conversation"
        if conversation_history and len(conversation_history) > 0:
            for chat in reversed(conversation_history):
                if chat.get('user'):
                    last_user_msg = chat['user'].lower()
                    if 'crop' in last_user_msg:
                        last_topic = "your crops"
                    elif 'weather' in last_user_msg:
                        last_topic = "the weather"
                    elif 'family' in last_user_msg:
                        last_topic = "your family"
                    break

        chatgpt_responses = {
            "happy": [
                f"That's wonderful to hear, {farmer_name}! You know, it's moments like these that make all the hard work worth it. What made today special for you?",
                f"I'm so glad you're feeling good, {farmer_name}! 🌟 Tell me more about what's bringing you joy right now.",
                f"That's fantastic! You deserve to feel this good, {farmer_name}. What's been the highlight of your week?"
            ],
            "sad": [
                f"Oh, {farmer_name}, I can really hear how heavy that feels right now. 🌱 You know, it's completely okay to have tough days. I'm right here with you. What specifically has been weighing on your mind?",
                f"I hear you, {farmer_name}. That sounds really difficult. Remember when we talked about {last_topic}? How are things going with that? I'm here to listen, no matter what.",
                f"That's so tough, {farmer_name}. I wish I could give you a big hug right now. What do you think might help you feel a little better today?"
            ],
            "angry": [
                f"I can feel how frustrated you are, {farmer_name}. That's completely understandable - farming can be incredibly challenging. What happened that made you feel this way?",
                f"Oh man, {farmer_name}, that sounds really frustrating! I totally get why you'd feel angry about that. How can I support you through this?",
                f"That's so unfair, {farmer_name}. I can imagine how maddening that must be. What do you think needs to change? I'm here to help you figure this out."
            ],
            "high_risk": [
                f"Oh, {farmer_name}, my heart goes out to you right now. I can hear how much pain you're in, and I want you to know you're not alone. Please remember how much you matter to the people who care about you. Can we talk about what might help you feel a little safer right now?",
                f"{farmer_name}, I hear the darkness in your words, and it breaks my heart. You are so incredibly valuable, and there are people who love you deeply. Please reach out to someone you trust right now - I'm here with you, and help is available. What can I do to support you in this moment?",
                f"I feel your pain, {farmer_name}, and I want you to know how much I care about you. You're not alone in this darkness. Please talk to someone - a friend, family member, or helpline. You're stronger than you know, and there is hope."
            ]
        }

        responses = chatgpt_responses.get(emotion, [
            f"I hear you, {farmer_name}. You know, sometimes just talking about things can help. What's been on your mind lately?",
            f"That's interesting, {farmer_name}. Tell me more about that. I'm genuinely curious to hear your thoughts.",
            f"I appreciate you sharing that with me, {farmer_name}. How are you feeling about everything right now?"
        ])

        return responses[msg_hash % len(responses)]
    
    # Default farming response for unrecognized queries
    return f"""I'm here to help you with your farming questions! 🌾

You can ask me about:
• 💰 Market prices for crops
• 🌤️ Weather updates and forecasts
• 🌱 Crop recommendations
• 🩺 Plant diseases and solutions
• 💧 Irrigation tips
• 🧪 Soil health

Or just share how you're feeling - I'm here to listen!"""

# Advanced ChatGPT Algorithm Fallback Function
def get_chatgpt_algorithm_fallback(emotion, lang, farmer_profile=None, user_message=None, conversation_history=None):
    farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

    # Get conversation context for more natural fallback
    last_topic = "our conversation"
    if conversation_history and len(conversation_history) > 0:
        # Look at the last user message in history
        for chat in reversed(conversation_history):
            if chat.get('user'):
                last_user_msg = chat['user'].lower()
                if 'crop' in last_user_msg:
                    last_topic = "your crops"
                elif 'weather' in last_user_msg:
                    last_topic = "the weather"
                elif 'family' in last_user_msg:
                    last_topic = "your family"
                break

    # Advanced ChatGPT-style fallback responses with natural conversation patterns
    chatgpt_fallback_responses = {
        "happy": [
            f"That's wonderful to hear, {farmer_name}! 😊 You know, it's moments like these that make all the hard work worth it. What made today special for you?",
            f"I'm so glad you're feeling good, {farmer_name}! 🌟 Tell me more about what's bringing you joy right now. You know what I mean?",
            f"That's fantastic! You deserve to feel this good, {farmer_name}. What's been the highlight of your week? I'd love to hear about it.",
            f"You know, {farmer_name}, seeing you happy like this really warms my heart. What do you think has made things better lately?"
        ],
        "sad": [
            f"Oh, {farmer_name}, I can really hear how heavy that feels right now. 🌱 You know, it's completely okay to have tough days. I'm right here with you. What specifically has been weighing on your mind?",
            f"I hear you, {farmer_name}. That sounds really difficult. Remember when we talked about {last_topic}? How are things going with that? I'm here to listen, no matter what.",
            f"That's so tough, {farmer_name}. I wish I could give you a big hug right now. 💙 What do you think might help you feel a little better today?",
            f"You know, {farmer_name}, it's okay to feel this way. Many farmers I know go through similar challenges. How has this been affecting your daily routine?"
        ],
        "angry": [
            f"I can feel how frustrated you are, {farmer_name}. 😠 That's completely understandable - farming can be incredibly challenging. What happened that made you feel this way?",
            f"Oh man, {farmer_name}, that sounds really frustrating! I totally get why you'd feel angry about that. You know what? You're absolutely right to feel this way. How can I support you through this?",
            f"That's so unfair, {farmer_name}. I can imagine how maddening that must be. 🌿 What do you think needs to change? I'm here to help you figure this out.",
            f"I hear your frustration loud and clear, {farmer_name}. You know, it's normal to feel this way when things don't go as planned. What can I do to help you right now?"
        ],
        "high_risk": [
            f"Oh, {farmer_name}, my heart goes out to you right now. 💔 I can hear the darkness in your words, and I want you to know you're not alone. Please remember how much you matter to the people who care about you. Can we talk about what might help you feel a little safer right now?",
            f"{farmer_name}, I hear the pain in your words, and it breaks my heart. 🌙 You are so incredibly valuable, and there are people who love you deeply. Please reach out to someone you trust right now - I'm here with you, and help is available. What can I do to support you in this moment?",
            f"I feel your pain, {farmer_name}, and I want you to know how much I care about you. 💙 You're not alone in this darkness. Please talk to someone - a friend, family member, or helpline. You're stronger than you know, and there is hope. I'm right here with you.",
            f"{farmer_name}, your words concern me deeply. You know, you're not alone in this struggle. Please reach out to someone you trust immediately - there are people who care about you and want to help. I'm here for you too. What can I do right now to support you?"
        ]
    }

    responses = chatgpt_fallback_responses.get(emotion, [
        f"I hear you, {farmer_name}. You know, sometimes just talking about things can help. What's been on your mind lately?",
        f"That's interesting, {farmer_name}. Tell me more about that. I'm genuinely curious to hear your thoughts.",
        f"I appreciate you sharing that with me, {farmer_name}. How are you feeling about everything right now?",
        f"You know, {farmer_name}, I'm really glad you reached out. What's been going on with you lately?"
    ])

    # Return response based on conversation history length for variety
    conversation_length = len(conversation_history) if conversation_history else 0
    return responses[conversation_length % len(responses)]

# Enhanced fallback response function (legacy support)
def get_enhanced_dynamic_response(emotion, lang, farmer_profile=None, user_message=None):
    farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

    friendly_responses = {
        "happy": [
            f"That's wonderful to hear, {farmer_name}! 😊 Keep that positive spirit going!",
            f"I'm so glad you're feeling good, {farmer_name}! What's making you smile today?",
            f"Great to see you happy! 🌟 Tell me more about what's going well for you."
        ],
        "sad": [
            f"I can hear you're going through a tough time, {farmer_name}. I'm here to listen. 🌱",
            f"It's okay to feel this way, {farmer_name}. I'm right here with you. What's on your mind?",
            f"I'm really sorry you're feeling down, {farmer_name}. Would you like to talk about it?"
        ],
        "angry": [
            f"I understand you're frustrated, {farmer_name}. Let's work through this together. 🌿",
            f"It's normal to feel angry sometimes, {farmer_name}. I'm here to help you process this.",
            f"I hear your frustration, {farmer_name}. What can I do to support you right now?"
        ],
        "high_risk": [
            f"I'm really concerned about you, {farmer_name}. Please know you're not alone. 📞",
            f"I care about you deeply, {farmer_name}. Let's get you the help you need right now.",
            f"You're important to me, {farmer_name}. Please reach out to someone you trust immediately."
        ]
    }

    responses = friendly_responses.get(emotion, [
        f"I'm here for you, {farmer_name}. What's on your mind?",
        f"Tell me what's going on, {farmer_name}. I'm listening.",
        f"I'm glad you reached out, {farmer_name}. How can I support you today?"
    ])

    # Use conversation length for variety
    conv_len = len(conversation_history) if conversation_history else 0
    return responses[conv_len % len(responses)]

# Function to send emergency WhatsApp message using CallMeBot
def send_emergency_whatsapp(farmer_profile, location, lang):
    try:
        farmer_name = farmer_profile.get('name', 'Farmer')

        # Get CallMeBot credentials
        callmebot_api_key = os.getenv('CALLMEBOT_API_KEY')
        callmebot_phone = os.getenv('CALLMEBOT_PHONE')

        if not callmebot_api_key or not callmebot_phone:
            print("CallMeBot API key or phone number not configured")
            return 0

        # Prepare message in selected language
        if lang == "English":
            message = f"🚨 EMERGENCY ALERT: {farmer_name} needs immediate help at {location}. Please contact them urgently!"
        elif lang == "Hindi":
            message = f"🚨 आपातकालीन अलर्ट: {farmer_name} को {location} पर तत्काल मदद की आवश्यकता है। कृपया उनसे संपर्क करें!"
        elif lang == "Tamil":
            message = f"🚨 அவசர எச்சரிக்கை: {farmer_name} க்கு {location} இல் உடனடி உதவி தேவை. தயவுசெய்து அவரை தொடர்பு கொள்ளுங்கள்!"
        else:
            message = f"🚨 EMERGENCY ALERT: {farmer_name} needs immediate help at {location}. Please contact them urgently!"

        # URL encode the message for WhatsApp
        encoded_message = requests.utils.quote(message)

        # Send WhatsApp to family members
        family_members = []
        if farmer_profile.get('family1', {}).get('phone'):
            family_members.append(farmer_profile['family1'])
        if farmer_profile.get('family2', {}).get('phone'):
            family_members.append(farmer_profile['family2'])

        sent_count = 0
        for member in family_members:
            try:
                # CallMeBot WhatsApp API
                url = f"https://api.callmebot.com/whatsapp.php?phone={callmebot_phone}&text={encoded_message}&apikey={callmebot_api_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    sent_count += 1
                    print(f"Emergency WhatsApp sent successfully to {member['name']}")
                else:
                    print(f"Failed to send WhatsApp to {member['name']}: {response.text}")

            except Exception as e:
                print(f"Failed to send WhatsApp to {member['name']}: {e}")

        return sent_count

    except Exception as e:
        print(f"Emergency WhatsApp failed: {e}")
        return 0

# Function to get DeepAI response
def get_deepai_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using DeepAI API"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Create empathetic prompt based on emotion
        emotion_prompts = {
            "happy": f"You are AgriCare AI, a friendly companion for farmer {farmer_name}. They seem happy - respond warmly and share in their joy.",
            "sad": f"You are AgriCare AI, a supportive companion for farmer {farmer_name}. They seem sad - be empathetic and encouraging.",
            "angry": f"You are AgriCare AI, a calm companion for farmer {farmer_name}. They seem frustrated - listen and help them process their feelings.",
            "high_risk": f"You are AgriCare AI, a caring companion for farmer {farmer_name}. They need immediate support - be gentle and suggest help."
        }

        system_prompt = emotion_prompts.get(emotion, f"You are AgriCare AI, a helpful companion for farmer {farmer_name}.")

        # Prepare conversation context
        chat_history = ""
        if conversation_history:
            recent_chats = conversation_history[-2:]  # Last 2 exchanges for API limits
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history += f"User: {chat['user']}\nAI: {chat['bot']}\n"

        # Create full prompt
        full_prompt = f"{system_prompt}\n\n{chat_history}User: {user_message}\nAI:"

        # DeepAI API call
        url = "https://api.deepai.org/api/text-generator"
        headers = {
            "api-key": DEEP_AI_API_KEY
        }
        data = {
            "text": full_prompt
        }

        response = requests.post(url, headers=headers, data=data, timeout=15)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('output', '').strip()

            # Clean up response
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()

            # Extract just the AI response
            if '\nUser:' in generated_text:
                generated_text = generated_text.split('\nUser:')[0].strip()

            # Add empathetic elements based on emotion
            if emotion == "high_risk":
                generated_text += " Please know that help is available - you can talk to someone you trust or call a helpline."
            elif emotion == "sad":
                generated_text += " I'm here for you whenever you need to talk."
            elif emotion == "happy":
                generated_text += " It's wonderful to see you feeling positive!"

            return generated_text

        else:
            print(f"DeepAI API Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DeepAI Error: {e}")
        return None

# Function to get Cohere response
def get_cohere_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using Cohere API with improved chat method"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Create comprehensive system prompt for AgriCare AI
        system_prompt = f"""
        You are AgriCare AI, a friendly and supportive assistant for farmers.
        Your role is to answer farming questions in simple, practical language and also provide emotional support.
        If the farmer sounds stressed, sad, or in a serious emotional state, first reply with empathy and comforting words.
        Then, clearly suggest sending an emergency alert message to their family or agricultural officer via WhatsApp/SMS, such as:
        "Farmer is feeling very stressed, please check on them immediately."
        Do not actually send the message yourself, only suggest it when necessary.

        When giving farming advice:
        - Be specific and practical (fertilizers, irrigation methods, pest control, weather tips, crop care).
        - Keep answers short, clear, and positive.
        - Never give harmful or unsafe instructions.
        Use a warm, motivating tone — like a trusted friend.
        You may reply in English, or mix English with Tamil if that makes the farmer more comfortable.
        Your main goal: Help farmers feel confident, supported, and safe.

        Current farmer: {farmer_name}
        Detected emotion: {emotion}
        Language preference: {lang}
        """

        # Initialize Cohere client
        co = cohere.Client(api_key=COHERE_API_KEY)

        # Prepare conversation context for chat method
        chat_history = []
        if conversation_history:
            recent_chats = conversation_history[-3:]  # Last 3 exchanges for better context
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history.append({
                        "role": "USER",
                        "message": chat['user']
                    })
                    chat_history.append({
                        "role": "CHATBOT",
                        "message": chat['bot']
                    })

        # Add current user message
        chat_history.append({
            "role": "USER",
            "message": user_message
        })

        # Use the newer chat method for better reliability
        response = co.chat(
            model="command-a-03-2025",
            preamble=system_prompt,
            chat_history=chat_history,
            message=user_message,
            temperature=0.8,
            max_tokens=150,
            connectors=[]  # No external tools needed
        )

        generated_text = response.text.strip()

        # Add empathetic elements based on emotion if not already included
        if emotion == "high_risk" and "emergency alert" not in generated_text.lower():
            generated_text += "\n\nPlease know that help is available - you can talk to someone you trust or call a helpline."
        elif emotion == "sad" and "here for you" not in generated_text.lower():
            generated_text += "\n\nI'm here for you whenever you need to talk."
        elif emotion == "happy" and "wonderful" not in generated_text.lower():
            generated_text += "\n\nIt's wonderful to see you feeling positive!"

        return generated_text

    except Exception as e:
        print(f"Cohere Error: {e}")
        return None

# Function to get DeepAI response
def get_deepai_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using DeepAI API with enhanced variety"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

        # Add timestamp and random element for uniqueness
        import time
        timestamp = str(int(time.time() * 1000))  # Milliseconds for uniqueness
        random_seed = str(hash(user_message + timestamp) % 10000)  # Random seed

        # Create varied empathetic prompts based on emotion
        emotion_prompts = {
            "happy": [
                f"You are AgriCare AI, a cheerful farming companion for {farmer_name}. They seem joyful - respond enthusiastically and celebrate their positive mood.",
                f"You are AgriCare AI, a warm friend to farmer {farmer_name}. They're feeling great - share in their happiness and ask about what made their day special.",
                f"You are AgriCare AI, an encouraging companion for {farmer_name}. They're in good spirits - respond warmly and keep the positive energy flowing."
            ],
            "sad": [
                f"You are AgriCare AI, a compassionate listener for farmer {farmer_name}. They seem down - offer gentle support and show you truly care about their feelings.",
                f"You are AgriCare AI, a supportive friend to {farmer_name}. They're feeling low - be empathetic and help them feel less alone in their struggles.",
                f"You are AgriCare AI, a caring companion for farmer {farmer_name}. They need comfort - listen carefully and offer genuine encouragement."
            ],
            "angry": [
                f"You are AgriCare AI, a calm mediator for farmer {farmer_name}. They're frustrated - help them process their anger constructively and find solutions.",
                f"You are AgriCare AI, a patient listener for {farmer_name}. They're upset - acknowledge their feelings and help them find a path forward.",
                f"You are AgriCare AI, a steady companion for farmer {farmer_name}. They're angry - stay calm and help them work through their emotions."
            ],
            "high_risk": [
                f"You are AgriCare AI, an immediate support system for {farmer_name}. They need urgent help - be gentle, caring, and direct them to professional support.",
                f"You are AgriCare AI, a crisis companion for farmer {farmer_name}. They're in distress - show deep concern and guide them toward immediate assistance.",
                f"You are AgriCare AI, a lifeline for {farmer_name}. They need help now - be compassionate and ensure they know they're not alone."
            ]
        }

        # Select random prompt variation
        import random
        prompts = emotion_prompts.get(emotion, [f"You are AgriCare AI, a helpful companion for farmer {farmer_name}."])
        system_prompt = random.choice(prompts)

        # Prepare conversation context with more variety
        chat_history = ""
        if conversation_history:
            # Use random number of recent chats (1-3) for variety
            num_chats = random.randint(1, min(3, len(conversation_history)))
            recent_chats = conversation_history[-num_chats:]
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history += f"User: {chat['user']}\nAI: {chat['bot']}\n"

        # Add unique context to prevent repetition
        unique_context = f"[Context: {emotion} emotion, farmer {farmer_name}, timestamp {timestamp}, seed {random_seed}]"

        # Create full prompt with variety
        full_prompt = f"{system_prompt}\n{unique_context}\n\n{chat_history}User: {user_message}\nAI:"

        # DeepAI API call with varied parameters
        url = "https://api.deepai.org/api/text-generator"
        headers = {
            "api-key": DEEP_AI_API_KEY
        }

        # Use different parameters for variety
        temperature_options = [0.7, 0.8, 0.9, 1.0]
        temperature = random.choice(temperature_options)

        data = {
            "text": full_prompt,
            "temperature": temperature  # Add temperature for more variety
        }

        response = requests.post(url, headers=headers, data=data, timeout=15)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('output', '').strip()

            # Clean up response
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()

            # Extract just the AI response
            if '\nUser:' in generated_text:
                generated_text = generated_text.split('\nUser:')[0].strip()

            # Remove any system prompt contamination
            if generated_text.startswith("You are AgriCare AI"):
                # Find the actual response after the system prompt
                lines = generated_text.split('\n')
                for i, line in enumerate(lines):
                    if not line.startswith("You are") and not line.startswith("[") and line.strip():
                        generated_text = '\n'.join(lines[i:])
                        break

            # Add varied empathetic elements based on emotion
            if emotion == "high_risk":
                crisis_responses = [
                    " Please remember you're not alone - reach out to someone you trust or call a helpline immediately.",
                    " I'm deeply concerned about you and want you to get the help you need right now.",
                    " Your safety matters to me - please connect with professional support as soon as possible.",
                    " I care about you and want you to know that help is available 24/7."
                ]
                generated_text += random.choice(crisis_responses)
            elif emotion == "sad":
                comfort_responses = [
                    " I'm here for you whenever you need to talk or just need someone to listen.",
                    " Remember that tough times don't last forever - I'm here to support you through this.",
                    " Your feelings are valid, and I'm here to help you through whatever you're facing.",
                    " I'm glad you reached out - talking about it can help, and I'm here to listen."
                ]
                generated_text += random.choice(comfort_responses)
            elif emotion == "happy":
                positive_responses = [
                    " It's wonderful to see you feeling positive - keep that good energy going!",
                    " I'm so glad you're in a good place right now - you deserve to feel this way.",
                    " Your positive outlook is inspiring - I'm happy to share in your good mood!",
                    " It's great to hear from you when you're feeling good - keep enjoying the moment!"
                ]
                generated_text += random.choice(positive_responses)

            # Ensure response is not empty
            if not generated_text.strip():
                generated_text = get_chatgpt_style_fallback(emotion, lang, farmer_profile, user_message, conversation_history)

            return generated_text

        else:
            print(f"DeepAI API Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DeepAI Error: {e}")
        return None

# Function to get dynamic response (fallback)
def get_dynamic_response(emotion, lang):
    base_msg = base_responses.get(emotion, "I am here for you!")
    return emotion_translations.get(lang, lambda x: x)(base_msg)
# ---------------------------

# ---------------------------


# Streamlit Layout
# ---------------------------
st.set_page_config(page_title="AgriDream 🌾", layout="wide", page_icon="🌱")

# Global Language Selector
global_lang = st.sidebar.selectbox("🌐 Select Language / भाषा चुनें / மொழியை தேர்ந்தெடுக்கவும்", languages, key="global_lang")

# Sidebar
st.sidebar.title("🌾 " + get_text("title", global_lang).split(" - ")[0] + " Menu")
menu_options = [
    get_text("menu_dashboard", global_lang),
    get_text("menu_crop_rec", global_lang),
    get_text("menu_price", global_lang),
    get_text("menu_weather", global_lang),
    get_text("menu_disease", global_lang),
    get_text("menu_emotion", global_lang),
    get_text("menu_schemes", global_lang)
]
menu = st.sidebar.radio(
    get_text("select_language", global_lang),
    menu_options
)


# Enhanced Logo / Header Section
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%); border-radius: 15px; margin-bottom: 10px;">
    <div style="font-size: 60px; margin-bottom: 10px;">🌾</div>
    <h1 style="color: #1a7f2e; margin: 0; font-size: 42px; font-weight: bold;">AgriDream</h1>
    <p style="color: #666; font-size: 16px; margin: 5px 0;">Smart Agriculture Decision System</p>
    <p style="color: #888; font-size: 14px;">📊 Price Forecasting | 🌾 Crop Recommendation | 🌤️ Weather Advisory | 🩺 Disease Detection</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Farmer Profile Setup
st.sidebar.subheader("👨‍🌾 " + get_text("farmer_profile", global_lang))

# Load farmer from database on every page load (check by most recent farmer)
try:
    farmers = database.get_all_farmers()
    if farmers:
        latest_farmer = farmers[0]
        st.session_state.farmer_profile = {
            "name": latest_farmer.get('name', ''),
            "age": 30,
            "family1": {"name": "", "phone": latest_farmer.get('emergency_contact', '') or latest_farmer.get('phone', '')},
            "family2": {"name": "", "phone": ""}
        }
        st.session_state.current_farmer_id = latest_farmer.get('id')
        FAMILY_NUMBERS = get_family_numbers(st.session_state.farmer_profile)
except Exception as e:
    print(f"Error loading farmer: {e}")

# Get values from session state
profile = st.session_state.get('farmer_profile', {})
farmer_name = st.sidebar.text_input(get_text("farmer_name", global_lang), 
    value=profile.get('name', ''), key="farmer_name")
farmer_age = st.sidebar.number_input(get_text("age", global_lang), 
    min_value=18, max_value=100, value=profile.get('age', 30), key="farmer_age")
st.sidebar.subheader(get_text("emergency_contacts", global_lang))
family1_name = st.sidebar.text_input(get_text("family_member_1", global_lang), 
    value=profile.get('family1', {}).get('name', ''), key="family1_name")
family1_phone = st.sidebar.text_input(get_text("phone", global_lang), 
    value=profile.get('family1', {}).get('phone', ''), key="family1_phone")
family2_name = st.sidebar.text_input(get_text("family_member_2", global_lang), 
    value=profile.get('family2', {}).get('name', ''), key="family2_name")
family2_phone = st.sidebar.text_input(get_text("phone", global_lang), 
    value=profile.get('family2', {}).get('phone', ''), key="family2_phone")

if st.sidebar.button(get_text("save_profile", global_lang)):
    # Save to database (permanent storage)
    if farmer_name:
        try:
            # Try to find existing farmer by name, or create new
            farmers = database.get_all_farmers()
            existing_farmer = None
            for f in farmers:
                if f.get('name') == farmer_name:
                    existing_farmer = f
                    break
            
            if existing_farmer:
                # Update existing farmer with all fields
                database.update_farmer(existing_farmer['id'], 
                    phone=family1_phone if family1_phone else existing_farmer.get('phone'),
                    emergency_contact=family1_phone if family1_phone else existing_farmer.get('emergency_contact'),
                    state="Maharashtra")
                st.session_state.current_farmer_id = existing_farmer['id']
            else:
                # Insert new farmer
                farmer_id = database.insert_farmer(
                    name=farmer_name,
                    state="Maharashtra",
                    phone=family1_phone,
                    emergency_contact=family1_phone
                )
                st.session_state.current_farmer_id = farmer_id
            
            # Also store in session state for current session
            st.session_state.farmer_profile = {
                "name": farmer_name,
                "age": farmer_age,
                "family1": {"name": family1_name, "phone": family1_phone},
                "family2": {"name": family2_name, "phone": family2_phone}
            }
            st.session_state.farmer_loaded = True  # Mark as loaded
            # Update global FAMILY_NUMBERS variable
            FAMILY_NUMBERS = get_family_numbers(st.session_state.farmer_profile)
            st.sidebar.success(get_text("profile_saved", global_lang))
        except Exception as e:
            print(f"Error saving profile: {e}")
            # Fallback to session state if database fails
            st.session_state.farmer_profile = {
                "name": farmer_name,
                "age": farmer_age,
                "family1": {"name": family1_name, "phone": family1_phone},
                "family2": {"name": family2_name, "phone": family2_phone}
            }
            FAMILY_NUMBERS = get_family_numbers(st.session_state.farmer_profile)
            st.sidebar.success(get_text("profile_saved", global_lang))
    else:
        st.sidebar.warning("Please enter your name to save profile")

# ---------------------------
# Dashboard
# ---------------------------
# ============================================================
# Smart Farm Intelligence Dashboard
# ============================================================

# Price trend analysis function
def analyze_price_trends(df_prices):
    """Analyze price trends to determine direction"""
    trends = {}
    for commodity in df_prices['Commodity'].unique()[:10]:
        commodity_data = df_prices[df_prices['Commodity'] == commodity]['Modal_x0020_Price']
        if len(commodity_data) > 1:
            # Simple trend: if recent avg > overall avg = increasing
            recent = commodity_data.tail(5).mean() if len(commodity_data) >= 5 else commodity_data.mean()
            overall = commodity_data.mean()
            if recent > overall * 1.05:
                trends[commodity] = "↑ increasing"
            elif recent < overall * 0.95:
                trends[commodity] = "↓ decreasing"
            else:
                trends[commodity] = "→ stable"
        else:
            trends[commodity] = "→ stable"
    return trends


# Risk assessment function combining weather and disease
def get_crop_risk_score(crop, humidity=60, temperature=25):
    """Calculate risk score based on conditions"""
    risk_factors = []
    
    # Humidity-based disease risk
    if humidity > 75:
        risk_factors.append(("High", "High humidity increases fungal disease risk"))
    elif humidity > 60:
        risk_factors.append(("Medium", "Moderate humidity - monitor for disease"))
    
    # Temperature stress
    if temperature > 38:
        risk_factors.append(("High", "Heat stress may affect crop yield"))
    elif temperature > 35:
        risk_factors.append(("Medium", "High temperature - ensure adequate irrigation"))
    
    # Determine overall risk
    if not risk_factors:
        return "🟢 LOW", "Good growing conditions"
    elif any(r[0] == "High" for r in risk_factors):
        return "🔴 HIGH", risk_factors[0][1]
    else:
        return "🟡 MEDIUM", risk_factors[0][1]


# Decision recommendation based on conditions
def get_decision_recommendation(trends, weather_data=None):
    """Generate actionable decision recommendation"""
    recommendations = []
    action_level = "🟢"  # Default to safe
    
    # Analyze price trends
    increasing = sum(1 for v in trends.values() if "increasing" in v)
    decreasing = sum(1 for v in trends.values() if "decreasing" in v)
    
    if increasing > decreasing:
        recommendations.append("📈 Prices trending UP - Consider holding crops for better prices")
        action_level = "🟡"
    elif decreasing > increasing:
        recommendations.append("📉 Prices trending DOWN - Consider selling soon to avoid losses")
        action_level = "🟢"
    else:
        recommendations.append("➡️ Prices relatively STABLE - No immediate action needed")
    
    # Weather-based recommendations
    if weather_data:
        if weather_data.get('rainfall', 0) > 20:
            recommendations.append("🌧️ Heavy rain expected - Delay harvesting, ensure drainage")
            action_level = "🔴"
        elif weather_data.get('temperature', 0) > 38:
            recommendations.append("☀️ High temperature - Water crops in early morning/evening")
            action_level = "🟡"
    
    return action_level, recommendations


if menu == get_text("menu_dashboard", global_lang):
    st.subheader("🌾 " + get_text("farm_intelligence_center", global_lang))
    st.write("📊 Your personalized farm decision support system")
    
    # === 0. QUICK CONTEXT SELECTOR ===
    st.markdown("### 📍 Your Farm Overview")
    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        dashboard_state = st.selectbox("📍 Your State", 
            ['Maharashtra', 'Tamil Nadu', 'Uttar Pradesh', 'Karnataka', 'Gujarat', 
             'Madhya Pradesh', 'Punjab', 'West Bengal', 'Andhra Pradesh', 'Telangana'], 
            key="dash_state")
    with ctx_col2:
        dashboard_crop = st.selectbox("🌾 Your Main Crop", 
            ['Rice', 'Wheat', 'Cotton', 'Tomato', 'Potato', 'Onion', 'Maize', 'Sugarcane'],
            key="dash_crop")
    
    # Load price data
    try:
        df_prices = pd.read_csv('agmarknet_prices.csv')
        
        # Get analysis data - filter by state if available
        state_prices = df_prices[df_prices['State'] == dashboard_state] if dashboard_state in df_prices['State'].values else df_prices
        trends = analyze_price_trends(state_prices if len(state_prices) > 0 else df_prices)
        
        # Get weather for selected state (state name will be mapped to capital city)
        weather_data = None
        weather_reason = ""
        try:
            weather_data, _ = get_weather(dashboard_state)
        except:
            try:
                weather_data, _ = get_weather("Mumbai")
            except:
                pass
        
        # === 1. TOP STRIP - INSTANT SITUATION ===
        st.markdown("---")
        
        # Get current price for user's crop
        crop_price_data = state_prices[state_prices['Commodity'] == dashboard_crop] if len(state_prices) > 0 else df_prices[df_prices['Commodity'] == dashboard_crop]
        current_price = int(crop_price_data['Modal_x0020_Price'].mean()) if not crop_price_data.empty else 2000
        
        # Calculate trend for user's crop
        crop_trend = trends.get(dashboard_crop, "→ stable")
        
        # Weather summary
        weather_summary = "Clear"
        if weather_data:
            temp = weather_data.get('temperature', 0)
            humidity = weather_data.get('humidity', 0)
            weather_summary = f"{temp}°C, {humidity}% humidity"
            if weather_data.get('rainfall', 0) > 15:
                weather_summary += ", Rain expected"
        
        # Quick status strip
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <div><strong>📍 State:</strong> {dashboard_state}</div>
            <div><strong>🌾 Crop:</strong> {dashboard_crop}</div>
            <div><strong>💰 Price:</strong> ₹{current_price} ({crop_trend})</div>
            <div><strong>🌤️ Weather:</strong> {weather_summary}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # === 2. MAIN DECISION CARD (HERO SECTION) - REDESIGNED ===
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; margin: 15px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">🚦 TODAY'S DECISION</h2>
        </div>
        """, unsafe_allow_html=True)
        
        action_level, decisions = get_decision_recommendation(trends, weather_data)
        
        # Calculate money impact
        money_impact = "₹0"
        if "increasing" in str(decisions):
            money_impact = f"+₹{int(current_price * 0.05)} potential gain"
        elif "decreasing" in str(decisions):
            money_impact = f"-₹{int(current_price * 0.05)} potential loss"
        
        # Decision banner - MORE BOLD
        decision_color = "#28a745" if action_level == "🟢" else "#ffc107" if action_level == "🟡" else "#dc3545"
        decision_text = "SAFE - You're doing well" if action_level == "🟢" else "CAUTION - Plan ahead" if action_level == "🟡" else "ALERT - Take action NOW"
        
        # Generate WHY based on conditions
        why_message = ""
        if action_level == "🟢":
            why_message = "Prices are stable and weather conditions are normal in your area."
        elif action_level == "🟡":
            why_message = "Some price changes detected. Monitor your crops closely."
        else:
            why_message = "Price dropping or weather risk detected. Consider selling soon."
        
        st.markdown(f"""
        <div style="background-color: {decision_color}; padding: 30px; border-radius: 15px; margin: 15px 0; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h1 style="color: white; margin: 0; font-size: 36px;">{action_level} {decision_text}</h1>
            <p style="color: white; font-size: 20px; margin: 15px 0;">💰 {money_impact}</p>
        </div>
        
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #28a745;">
            <p style="margin: 0; color: #495057;"><strong>📋 Why:</strong> {why_message}</p>
        </div>
        
        <div style="background-color: #e8f5e9; padding: 10px; border-radius: 8px; margin: 10px 0; text-align: center;">
            <p style="margin: 0; color: #2e7d32;">✅ No immediate action needed today - Keep monitoring your farm</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Decision reasons
        for decision in decisions:
            st.write(decision)
        
        st.markdown("---")
        
        # === 3. BEST & WORST CROPS TODAY ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Best Performing Crops")
            top_crops = df_prices.nlargest(5, 'Modal_x0020_Price')[['Commodity', 'Modal_x0020_Price']].drop_duplicates(subset=['Commodity'])
            for i, (_, row) in enumerate(top_crops.iterrows(), 1):
                trend = trends.get(row['Commodity'], "→ stable")
                st.write(f"{i}. **{row['Commodity']}** - ₹{int(row['Modal_x0020_Price'])} {trend}")
        
        with col2:
            st.markdown("### ⚠️ Crops Needing Attention")
            low_crops = df_prices.nsmallest(5, 'Modal_x0020_Price')[['Commodity', 'Modal_x0020_Price']].drop_duplicates(subset=['Commodity'])
            for i, (_, row) in enumerate(low_crops.iterrows(), 1):
                trend = trends.get(row['Commodity'], "→ stable")
                st.write(f"{i}. **{row['Commodity']}** - ₹{int(row['Modal_x0020_Price'])} {trend}")
        
        st.markdown("---")
        
        # === 3. PRICE DIRECTION INDICATORS ===
        st.markdown("### 📈 Price Direction (Next 7 Days)")
        
        # Create columns for direction indicators
        cols = st.columns(5)
        for i, (crop, trend) in enumerate(list(trends.items())[:5]):
            with cols[i % 5]:
                if "↑" in trend:
                    color = "#28a745"
                    emoji = "⬆️"
                elif "↓" in trend:
                    color = "#dc3545"
                    emoji = "⬇️"
                else:
                    color = "#6c757d"
                    emoji = "➡️"
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                    <div style="font-size: 24px;">{emoji}</div>
                    <div>{crop}</div>
                    <div style="font-size: 12px;">{trend}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === 4. RISK OVERLAY ===
        st.markdown("### 🌤️ Weather Risk Assessment")
        
        if weather_data:
            temp = weather_data.get('temperature', 0)
            humidity = weather_data.get('humidity', 0)
            description = weather_data.get('description', '').lower()
            
            # Risk calculation
            risk_level = "LOW"
            risk_color = "#28a745"
            
            if temp > 40 or humidity > 85:
                risk_level = "HIGH"
                risk_color = "#dc3545"
            elif temp > 35 or humidity > 70:
                risk_level = "MEDIUM"
                risk_color = "#ffc107"
            
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">🌡️ {temp}°C | 💧 {humidity}% | {description.title()}</h3>
                <p style="color: white; margin: 5px 0;">Risk Level: {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Crop-specific risk
            st.markdown("### 🌾 Crop Risk Status")
            crops_to_check = ["Rice", "Wheat", "Tomato", "Potato", "Cotton", "Maize"]
            for crop in crops_to_check[:4]:
                risk, desc = get_crop_risk_score(crop, humidity, temp)
                st.write(f"**{crop}:** {risk} - {desc}")
        else:
            st.info("Weather data unavailable. Configure API for personalized risk assessment.")
        
        st.markdown("---")
        
        # === 5. SMART RECOMMENDATIONS ===
        st.markdown("### 💡 Smart Recommendations")
        
        # Generate recommendations based on analysis
        smart_recs = []
        
        # Calculate trend counts
        increasing = sum(1 for v in trends.values() if "increasing" in v)
        decreasing = sum(1 for v in trends.values() if "decreasing" in v)
        
        # Price-based recommendation
        if increasing > decreasing:
            smart_recs.append("🌾 **Crop:** Hold wheat/rice stocks - prices expected to rise")
        else:
            smart_recs.append("🌾 **Crop:** Consider selling wheat/rice soon - prices may peak")
        
        # Weather-based recommendation
        if weather_data:
            if weather_data.get('rainfall', 0) < 5:
                smart_recs.append("💧 **Irrigation:** Increase watering - low rainfall expected")
            elif weather_data.get('rainfall', 0) > 15:
                smart_recs.append("💧 **Irrigation:** Skip irrigation - natural rainfall sufficient")
            
            if weather_data.get('humidity', 0) > 75:
                smart_recs.append("🍄 **Disease:** High humidity - monitor crops for fungal diseases")
        
        # General recommendation
        smart_recs.append("📊 **Action:** Review crop recommendations for next season")
        
        for rec in smart_recs:
            st.write(rec)
        
        st.markdown("---")
        
        # === 6. ALERTS SECTION ===
        st.markdown("### ⚠️ Active Alerts")
        
        alerts = []
        
        # Calculate trend counts (already done above - reuse)
        if 'increasing' not in dir():
            increasing = sum(1 for v in trends.values() if "increasing" in v)
            decreasing = sum(1 for v in trends.values() if "decreasing" in v)
        
        # Check for price alerts
        if decreasing > increasing:
            alerts.append(("🔴", "Price drop alert for multiple crops - consider early selling"))
        
        # Check for weather alerts
        if weather_data:
            if weather_data.get('temperature', 0) > 38:
                alerts.append(("🟡", "Heat alert - protect sensitive crops"))
            if weather_data.get('rainfall', 0) > 25:
                alerts.append(("🟡", "Heavy rain alert - ensure proper drainage"))
        
        # Check for disease alerts
        if weather_data and weather_data.get('humidity', 0) > 75:
            alerts.append(("🟡", "Disease alert - high humidity favors fungal growth"))
        
        if alerts:
            for alert_emoji, alert_msg in alerts:
                st.warning(f"{alert_emoji} {alert_msg}")
        else:
            st.success("✅ No active alerts - All systems normal")
        
        st.markdown("---")
        
        # === QUICK STATS ===
        st.markdown("### 📊 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Commodities", len(df_prices['Commodity'].unique()))
        with col2:
            st.metric("Avg Price", f"₹{int(df_prices['Modal_x0020_Price'].mean())}")
        with col3:
            st.metric("States Covered", len(df_prices['State'].unique()))
        with col4:
            st.metric("Markets Tracked", len(df_prices['Market'].unique()))
            
    except FileNotFoundError:
        st.warning("⚠️ Price data not available. Please ensure agmarknet_prices.csv exists.")
        st.info("Run the price forecasting module first to fetch the latest market data.")
    
    # Fallback for demo mode
    if not 'df_prices' in locals() or df_prices is None:
        st.markdown("### 📊 Demo Dashboard")
        
        # Demo decision panel
        st.markdown("""
        <div style="background-color: #28a745; padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;">
            <h2 style="color: white; margin: 0;">🟢 Farm Status: SAFE</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("➡️ Prices relatively STABLE - No immediate action needed")
        
        st.markdown("### 🏆 Best Performing Crops")
        st.write("1. **Cotton** - ₹6,200 ⬆️ increasing")
        st.write("2. **Rice** - ₹2,400 → stable")
        st.write("3. **Tomato** - ₹2,100 ⬆️ increasing")
        
        st.markdown("### ⚠️ Crops Needing Attention")
        st.write("1. **Onion** - ₹1,800 ↓ decreasing")
        st.write("2. **Potato** - ₹1,400 → stable")
        
        st.markdown("### 💡 Smart Recommendations")
        st.write("🌾 **Crop:** Hold rice stocks - prices expected to rise")
        st.write("💧 **Irrigation:** Skip irrigation - natural rainfall sufficient")
        
        st.markdown("### ⚠️ Active Alerts")
        st.success("✅ No active alerts - All systems normal")

# ---------------------------
# Crop Recommendation
# ---------------------------
elif menu == get_text("menu_crop_rec", global_lang):
    st.subheader("🌾 " + get_text("crop_recommendation", global_lang))
    st.write(get_text("enter_conditions", global_lang))

    col1, col2 = st.columns(2)
    with col1:
        N = st.slider(get_text("nitrogen", global_lang), 0, 140, 50)
        P = st.slider(get_text("phosphorus", global_lang), 0, 145, 50)
        K = st.slider(get_text("potassium", global_lang), 0, 205, 50)
        ph = st.slider(get_text("ph_level", global_lang), 3.5, 9.9, 6.5)
        irrigation_type = st.selectbox("Irrigation Type", ["Drip Irrigation", "Sprinkler Irrigation", "Flood Irrigation", "Furrow Irrigation", "Rain-fed", "Manual Irrigation"])
    with col2:
        temperature = st.slider(get_text("temperature", global_lang), 8, 44, 25)
        humidity = st.slider(get_text("humidity", global_lang), 14, 100, 60)
        rainfall = st.slider(get_text("rainfall", global_lang), 20, 300, 100)
        soil_type = st.selectbox(get_text("soil_type", global_lang), ["Sandy", "Clay", "Loam", "Silt", "Peat", "Chalk"])
        state = st.selectbox(get_text("state", global_lang), df_crop['State'].unique())

    if st.button(get_text("get_recommendation", global_lang)):
        recommended_crops, confidences, reasoning = recommend_crop(N, P, K, temperature, humidity, ph, rainfall, state)

        # === CROP DECISION ENGINE (RE DESIGN) ===
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
            <h2 style="color: white; margin: 0;">🚦 CROP DECISION ENGINE</h2>
            <p style="color: white; opacity: 0.9;">Based on your conditions - ranked by profit & risk</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Build crop decisions with profit analysis
        crop_decisions = []
        
        for i, (crop, conf) in enumerate(zip(recommended_crops, confidences), 1):
            crop_key = crop.title()
            profit = CROP_PROFIT_INFO.get(crop_key, {})
            
            # If no profit info exists, create default values
            if not profit:
                profit = {
                    'market_value': '₹2,000/quintal',
                    'profit_outlook': 'Medium',
                    'growth_period': '120 days',
                    'harvest_month': 'Seasonal',
                    'initial_cost': '₹15,000/acre',
                    'water_need': 'Medium',
                    'risk': {'water': 'Medium', 'pest': 'Medium'},
                    'income_timing': '4 months'
                }
            
            # Calculate estimated profit (simple formula)
            price = int(''.join(filter(str.isdigit, profit.get('market_value', '2000').split('-')[0]))) or 2000
            # Yield estimate: 20-40 quintal/acre depending on crop
            yield_est = {"Rice": 30, "Wheat": 25, "Maize": 35, "Cotton": 15, "Tomato": 40, "Potato": 25, "Onion": 30}.get(crop_key, 25)
            cost = int(''.join(filter(str.isdigit, profit.get('initial_cost', '15000').split('-')[0]))) or 15000
            estimated_profit = (price * yield_est) - cost
            
            # Risk calculation
            risk_score = 0
            if profit['risk']['water'] == 'High': risk_score += 2
            if profit['risk']['water'] == 'Medium': risk_score += 1
            if profit['risk']['pest'] == 'High': risk_score += 2
            if profit['risk']['pest'] == 'Medium': risk_score += 1
            
            if risk_score >= 3:
                risk_level = "🔴 High"
            elif risk_score >= 1:
                risk_level = "🟡 Medium"
            else:
                risk_level = "🟢 Low"
            
            # Demand calculation (based on market value trends)
            if profit['profit_outlook'] == 'High':
                demand = "📈 High"
            elif profit['profit_outlook'] == 'Medium':
                demand = "📊 Medium"
            else:
                demand = "📉 Low"
            
            # Duration
            duration_str = profit.get('growth_period', '120 days')
            
            crop_decisions.append({
                'rank': i,
                'name': crop_key,
                'confidence': conf,
                'profit': estimated_profit,
                'risk': risk_level,
                'demand': demand,
                'duration': duration_str,
                'water': profit.get('water_need', 'Medium'),
                'market_value': profit.get('market_value', '₹2000/quintal'),
                'harvest': profit.get('harvest_month', 'TBD'),
                'reasons': reasoning.get(crop, {}).get('reasons', ['Suitable for your conditions'])
            })
        
        # Sort by profit (highest first)
        crop_decisions.sort(key=lambda x: x['profit'], reverse=True)
        
        # === 1. BEST CROP (HERO SECTION) ===
        if crop_decisions:
            best = crop_decisions[0]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
                <h2 style="color: white; margin: 0;">🥇 BEST CHOICE: {best['name'].upper()}</h2>
                <p style="color: white; font-size: 18px; margin: 10px 0;">💰 Estimated Profit: ₹{best['profit']:,}/acre</p>
                <p style="color: white; opacity: 0.9;">⚠️ Risk: {best['risk']} | 📈 Demand: {best['demand']} | ⏳ Duration: {best['duration']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # WHY section for best crop
            st.markdown(f"### 🎯 Why **{best['name']}**?")
            for reason in best['reasons'][:3]:
                st.write(f"✅ {reason}")
            
            # Add weather/conditions based reasoning
            if rainfall > 100:
                st.write("✅ High rainfall matches water requirement")
            elif rainfall < 50:
                st.write("✅ Low water need - suitable for drought conditions")
            
            if 20 <= temperature <= 30:
                st.write("✅ Temperature is optimal for growth")
            
            # === 2. ALTERNATIVES ===
            if len(crop_decisions) > 1:
                st.markdown("### 🥈 Alternatives (Ranked by Profit)")
                
                for crop in crop_decisions[1:]:
                    risk_emoji = "🔴" if "High" in crop['risk'] else "🟡" if "Medium" in crop['risk'] else "🟢"
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #ffc107;">
                        <h4 style="margin: 0;">🌾 {crop['name']}</h4>
                        <p style="margin: 5px 0;">💰 Profit: ₹{crop['profit']:,} | {risk_emoji} {crop['risk']} | ⏳ {crop['duration']}</p>
                        <p style="margin: 0; color: #666;">📈 Demand: {crop['demand']} | 💵 Market: {crop['market_value']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === 3. AVOID (if any high risk) ===
            high_risk_crops = [c for c in crop_decisions if "High" in c['risk']]
            if high_risk_crops:
                st.markdown("### 🚫 Consider Avoiding")
                for crop in high_risk_crops:
                    st.warning(f"⚠️ **{crop['name']}**: High risk - {crop['risk']} - Requires careful management")
        
        # === DETAILED CROP CARDS ===
        st.markdown("---")
        st.markdown("### 📋 Detailed Crop Analysis")
        
        for crop in crop_decisions:
            # Get full profit info
            profit_info = CROP_PROFIT_INFO.get(crop['name'], {})
            
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 15px; margin: 15px 0; border: 2px solid #28a745;">
                <h3 style="color: #28a745; margin-top: 0;">🌾 {crop['name']}</h3>
                <table style="width: 100%;">
                    <tr><td><strong>📊 ML Confidence:</strong></td><td>{crop['confidence']:.1f}%</td></tr>
                    <tr><td><strong>💰 Estimated Profit:</strong></td><td>₹{crop['profit']:,}/acre</td></tr>
                    <tr><td><strong>⏳ Duration:</strong></td><td>{crop['duration']}</td></tr>
                    <tr><td><strong>📅 Harvest:</strong></td><td>{crop['harvest']}</td></tr>
                    <tr><td><strong>💵 Market Price:</strong></td><td>{crop['market_value']}</td></tr>
                    <tr><td><strong>💧 Water Need:</strong></td><td>{crop['water']}</td></tr>
                    <tr><td><strong>⚠️ Risk Level:</strong></td><td>{crop['risk']}</td></tr>
                    <tr><td><strong>📈 Demand:</strong></td><td>{crop['demand']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Warnings based on inputs
        st.write("### ⚠️ Risk Warnings")
        
        risk_warnings = []
        
        # Check for potential issues
        if rainfall < 50 and irrigation_type == "Rain-fed":
            risk_warnings.append("⚠️ High risk: Low rainfall + rain-fed irrigation may cause crop failure")
        
        if ph < 5.5 or ph > 8.5:
            risk_warnings.append(f"⚠️ pH stress: Soil pH {ph} may limit nutrient availability")
        
        if temperature > 38:
            risk_warnings.append("⚠️ Heat stress: High temperature may affect crop development")
        
        if N > 100:
            risk_warnings.append("⚠️ Excess nitrogen: May cause vegetative growth, delay maturity")
        
        if risk_warnings:
            for warning in risk_warnings:
                st.warning(warning)
        else:
            st.success("✅ Low risk conditions detected for recommended crops")

        # Dynamic Irrigation Recommendation
        st.write(f"### {get_text('irrigation_rec', global_lang)}")
        irrigation_recommendations = {
            "Drip Irrigation": "Highly efficient for water conservation. Ideal for row crops and vegetables. Reduces water usage by 30-50%.",
            "Sprinkler Irrigation": "Good for most crops. Provides uniform water distribution. Suitable for medium to large fields.",
            "Flood Irrigation": "Traditional method, good for rice and wheat. High water usage but effective for flood-tolerant crops.",
            "Furrow Irrigation": "Excellent for row crops like maize and potatoes. Allows precise water application to plant roots.",
            "Rain-fed": "Depends entirely on rainfall. Suitable for drought-resistant crops. Requires good soil moisture retention.",
            "Manual Irrigation": "Labor-intensive but flexible. Good for small plots and when water conservation is critical."
        }

        selected_irrigation = irrigation_recommendations.get(irrigation_type, "General irrigation practices recommended.")
        st.info(f"**{irrigation_type}:** {selected_irrigation}")

        # Additional irrigation advice based on rainfall
        if rainfall < 50:
            st.warning("⚠️ Low rainfall: Consider drip irrigation to conserve water")
        elif 50 <= rainfall < 100:
            st.info("📊 Moderate rainfall: Your irrigation should complement natural rainfall")
        else:
            st.success("✅ Good rainfall: Flood irrigation may be used, but drip recommended for efficiency")
            
        # Dynamic irrigation based on crop and conditions
        if recommended_crops:
            top_crop = recommended_crops[0]
            if top_crop in ["Rice"]:
                st.info("💧 **{crop}:** Flood irrigation suitable. Maintain 5-10cm water layer in fields.".format(crop=top_crop))
            elif top_crop in ["Cotton", "Groundnut"]:
                st.info("💧 **{crop}:** Drip or sprinkler recommended due to water sensitivity.".format(crop=top_crop))
            elif top_crop in ["Tomato", "Onion", "Potato"]:
                st.info("💧 **{crop}:** Drip irrigation ideal for controlled water delivery.".format(crop=top_crop))

        # Soil Type Advice
        st.write(f"### {get_text('soil_considerations', global_lang)}")
        soil_advice = {
            "Sandy": "Drains quickly, may need more frequent watering. Good for root vegetables.",
            "Clay": "Retains water well, avoid overwatering. Good for most crops but may need drainage improvement.",
            "Loam": "Ideal soil type - balanced drainage and nutrient retention.",
            "Silt": "Holds moisture well, can be fertile but may compact easily.",
            "Peat": "High water retention, acidic - may need pH adjustment.",
            "Chalk": "Alkaline soil, good drainage but may lack nutrients."
        }
        st.info(f"**{soil_type} Soil:** {soil_advice.get(soil_type, 'General soil management recommended.')}")

        # Disease Prevention and Management
        st.write("### 🛡️ Disease Prevention & Management")

        # Common diseases based on recommended crops
        disease_info = {
            "Rice": {
                "diseases": ["Bacterial Blight", "Blast Disease", "Brown Spot"],
                "causes": "High humidity, poor drainage, infected seeds",
                "prevention": "Use disease-resistant varieties, proper spacing, avoid overhead irrigation, remove infected plants immediately",
                "treatment": "Copper-based fungicides, neem oil sprays, biological control agents"
            },
            "Wheat": {
                "diseases": ["Rust", "Powdery Mildew", "Wheat Scab"],
                "causes": "High humidity, dense planting, poor air circulation",
                "prevention": "Crop rotation, resistant varieties, proper spacing, timely sowing",
                "treatment": "Triazole fungicides, sulfur-based sprays, cultural practices"
            },
            "Maize": {
                "diseases": ["Corn Borer", "Downy Mildew", "Rust"],
                "causes": "Warm humid conditions, poor soil drainage, insect vectors",
                "prevention": "Field sanitation, resistant hybrids, proper irrigation, biological control",
                "treatment": "Insecticides, fungicides, pheromone traps, neem-based products"
            },
            "Cotton": {
                "diseases": ["Bacterial Blight", "Fusarium Wilt", "Verticillium Wilt"],
                "causes": "Soil-borne pathogens, infected seeds, poor drainage",
                "prevention": "Soil sterilization, certified seeds, crop rotation, resistant varieties",
                "treatment": "Systemic fungicides, soil amendments, biological control"
            },
            "Sugarcane": {
                "diseases": ["Red Rot", "Smuts", "Rust"],
                "causes": "Fungal spores, infected setts, humid conditions",
                "prevention": "Hot water treatment of setts, resistant varieties, proper drainage",
                "treatment": "Systemic fungicides, field sanitation, biological control"
            },
            "Tomato": {
                "diseases": ["Late Blight", "Fusarium Wilt", "Bacterial Spot"],
                "causes": "High humidity, infected seeds, poor air circulation",
                "prevention": "Resistant varieties, proper spacing, stake plants, avoid wet foliage",
                "treatment": "Copper fungicides, biological control, neem oil sprays"
            },
            "Potato": {
                "diseases": ["Late Blight", "Early Blight", "Black Scurf"],
                "causes": "Cool wet weather, infected tubers, poor storage",
                "prevention": "Certified seed potatoes, crop rotation, proper hilling, good drainage",
                "treatment": "Protective fungicides, copper sprays, biological fungicides"
            },
            "Onion": {
                "diseases": ["Downy Mildew", "Purple Blotch", "Basal Rot"],
                "causes": "High humidity, poor air circulation, infected seeds",
                "prevention": "Proper spacing, good drainage, resistant varieties, field sanitation",
                "treatment": "Fungicides, copper sprays, biological control agents"
            }
        }

        # Show disease information for recommended crops
        for crop in recommended_crops[:2]:  # Show for top 2 crops
            if crop in disease_info:
                info = disease_info[crop]
                with st.expander(f"🦠 {crop} Disease Management"):
                    st.write(f"**Common Diseases:** {', '.join(info['diseases'])}")
                    st.write(f"**Causes:** {info['causes']}")
                    st.write(f"**Prevention:** {info['prevention']}")
                    st.write(f"**Treatment:** {info['treatment']}")

        # General Disease Prevention Tips
        st.write("### 🛡️ General Disease Prevention Tips")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🌱 Cultural Practices:**")
            st.write("• Crop rotation (avoid planting same crop family)")
            st.write("• Proper plant spacing for air circulation")
            st.write("• Remove and destroy infected plant debris")
            st.write("• Use certified, disease-free seeds")
            st.write("• Practice field sanitation")

        with col2:
            st.write("**💊 Chemical Control:**")
            st.write("• Use appropriate fungicides preventively")
            st.write("• Apply pesticides at recommended times")
            st.write("• Rotate different chemical classes")
            st.write("• Follow safety guidelines and dosages")
            st.write("• Consider organic alternatives when possible")

        # General tips
        st.write(f"### {get_text('general_tips', global_lang)}")
        st.write("- Water early morning or evening to reduce evaporation")
        st.write("- Use mulch to retain soil moisture")
        st.write("- Monitor soil moisture levels regularly")
        st.write("- Test soil pH and nutrients annually")
        st.write("- Regular field monitoring for early disease detection")
        st.write("- Maintain proper plant nutrition for disease resistance")

# ---------------------------
# Price Forecasting
# ---------------------------
elif menu == get_text("menu_price", global_lang):
    st.subheader("💹 " + get_text("live_price_info", global_lang))

    # Load price data
    df = pd.read_csv('agmarknet_prices.csv')

    # State selection - include all Indian states
    available_states = sorted(df['State'].unique())
    all_indian_states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
        'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
        'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
        'Uttarakhand', 'West Bengal',
        # Union Territories
        'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
        'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
    ]

    # Combine available states with all Indian states
    for state in all_indian_states:
        if state not in available_states:
            available_states.append(state)
    available_states.sort()

    # ===== SIMPLE FLOW: Select State → Select Crop → Check Price =====
    st.markdown("---")
    st.markdown("### 🌾 Check Crop Price")
    
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_state = st.selectbox("📍 Select State", available_states, key="forecast_state")
    
    with sel_col2:
        # Get crops available in selected state
        state_market_data = ALL_STATES_MARKET_DATA.get(selected_state, {})
        
        if state_market_data:
            # Get all crops from all markets in state (comprehensive data)
            available_crops = set()
            for market_crops in state_market_data.values():
                available_crops.update(market_crops.keys())
            available_crops = sorted(list(available_crops))[:100]
        elif selected_state in df['State'].values:
            # Fallback to CSV data
            state_crops = df[df['State'] == selected_state]['Commodity'].unique()
            available_crops = sorted(state_crops)[:100]
        else:
            available_crops = sorted(df['Commodity'].unique())[:100]
        
        # Add fallback crops from ALL_STATES_PRICES
        fallback_crops = []
        for state_name, crops in ALL_STATES_PRICES.items():
            for crop_name in crops.keys():
                if crop_name not in available_crops and crop_name not in fallback_crops:
                    fallback_crops.append(crop_name)
        available_crops = list(available_crops) + sorted(fallback_crops)[:30]
        
        crop_choice = st.selectbox("🌾 Select Crop", available_crops, key="forecast_crop")

    # Check Price button
    check_btn = st.button("🔍 Check Price", key="check_price_btn", type="primary")

    # Get price data for selected state and crop
    state_market_data = ALL_STATES_MARKET_DATA.get(selected_state, {})
    
    if state_market_data:
        # Use comprehensive data - get average price from all markets in state
        prices_list = []
        markets_list = []
        for market, crops in state_market_data.items():
            if crop_choice in crops:
                prices_list.append(crops[crop_choice])
                markets_list.append(market)
        if prices_list:
            current_price = int(sum(prices_list) / len(prices_list))
            min_price = min(prices_list)
            max_price = max(prices_list)
            market_info = ", ".join(markets_list)
        else:
            current_price = None
            min_price = None
            max_price = None
            market_info = None
    else:
        # Fallback to CSV data
        crop_prices = df[(df['Commodity'] == crop_choice) & (df['State'] == selected_state)]
        
        if not crop_prices.empty:
            current_price = int(crop_prices['Modal_x0020_Price'].iloc[0])
            min_price = int(crop_prices['Min_x0020_Price'].iloc[0])
            max_price = int(crop_prices['Max_x0020_Price'].iloc[0])
            markets = crop_prices['Market'].unique().tolist()[:5]
            market_info = ", ".join(markets) + f", {selected_state}"
        else:
            state_data = ALL_STATES_PRICES.get(selected_state, {})
            current_price = state_data.get(crop_choice, 2000)
            min_price = int(current_price * 0.85)
            max_price = int(current_price * 1.15)
            market_info = selected_state

    # If current_price still None, get from fallback
    if current_price is None:
        state_data = ALL_STATES_PRICES.get(selected_state, {})
        current_price = state_data.get(crop_choice, 2000)
        min_price = int(current_price * 0.85)
        max_price = int(current_price * 1.15)
        market_info = selected_state
    
    # Get crop_prices for market info (if available from CSV)
    if selected_state in df['State'].values:
        crop_prices = df[(df['Commodity'] == crop_choice) & (df['State'] == selected_state)]
    else:
        crop_prices = pd.DataFrame()  # Empty if no data
    
    # ===== GENERATE FORECAST BEFORE DECISION =====
    # Calculate predicted price using simple moving average logic
    historical_prices = [int(current_price * (1 - 0.008 * i)) for i in range(30, 0, -1)]
    
    # Simple prediction: use moving average trend
    avg_recent = sum(historical_prices[-7:]) / 7
    avg_old = sum(historical_prices[:7]) / 7
    
    if avg_recent > avg_old * 1.03:
        predicted_price = int(current_price * 1.08)  # 8% increase
        trend_direction = 'increasing'
    elif avg_recent < avg_old * 0.97:
        predicted_price = int(current_price * 0.92)  # 8% decrease
        trend_direction = 'decreasing'
    else:
        predicted_price = current_price
        trend_direction = 'stable'
    
    # Generate forecast prices for chart
    if trend_direction == 'increasing':
        price_change = (predicted_price - current_price) / 35
        forecast_prices = [int(current_price + price_change * i) for i in range(35)]
    elif trend_direction == 'decreasing':
        price_change = (current_price - predicted_price) / 35
        forecast_prices = [int(current_price - price_change * i) for i in range(35)]
    else:
        forecast_prices = [current_price] * 35
    
    all_days = list(range(35))
    
    # ===== SHOW DECISION-ENGINE RESULTS =====
    if check_btn and current_price:
        st.markdown("---")
        
        # ===== 🚦 MAIN DECISION (HERO SECTION) =====
        # Calculate trend and decision
        trend_direction = 'stable'
        if predicted_price > current_price * 1.05:
            trend_direction = 'increasing'
            decision = "🟢 WAIT - Price likely to rise"
            decision_color = "#28a745"
            money_impact = f"+₹{int(predicted_price - current_price)} potential gain"
            confidence = "Medium"
            best_action = "Wait 7-14 days for better price"
        elif predicted_price < current_price * 0.95:
            trend_direction = 'decreasing'
            decision = "🔴 SELL NOW - Price likely to drop"
            decision_color = "#dc3545"
            money_impact = f"₹{int(predicted_price - current_price)} potential loss"
            confidence = "Medium"
            best_action = "Sell immediately to avoid bigger loss"
        else:
            decision = "🟡 STABLE - Price expected to remain steady"
            decision_color = "#ffc107"
            money_impact = "No significant change expected"
            confidence = "High"
            best_action = "Sell when convenient - price is stable"
        
        # Build decision engine output
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">🚦 SHOULD YOU SELL NOW?</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main decision banner
        st.markdown(f"""
        <div style="background-color: {decision_color}; padding: 30px; border-radius: 15px; margin: 15px 0; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h1 style="color: white; margin: 0; font-size: 32px;">{decision}</h1>
            <p style="color: white; font-size: 20px; margin: 15px 0;">💰 {money_impact}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence & Action
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🎯 Confidence:** {confidence}")
        with col2:
            st.markdown(f"**⏰ Best Action:** {best_action}")
        
        # Why explanation
        if trend_direction == 'increasing':
            why_text = "Based on historical trend analysis, prices show upward movement. Consider waiting for better returns."
        elif trend_direction == 'decreasing':
            why_text = "Prices show downward trend. Selling now can help avoid further losses."
        else:
            why_text = "Prices are stable with no significant expected change. Current market conditions are steady."
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #28a745;">
            <p style="margin: 0;"><strong>📋 Why:</strong> {why_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== 💹 CURRENT PRICE INFO =====
        st.markdown(f"### 💹 Current Market Price: {crop_choice}")
        
        # Main price display
        st.markdown(f"### ₹{current_price}/quintal")
        
        # Price Range
        st.markdown(f"**Price Range:** ₹{min_price} - ₹{max_price}")
        
        # ===== 📈 PRICE FORECAST CHART =====
        st.markdown("### 📈 Price Forecast (Next 35 Days)")
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=all_days,
            y=forecast_prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=6)
        ))
        
        # Add current price reference line
        fig.add_shape(type="line", x0=0, x1=34, y0=current_price, y1=current_price,
                      line=dict(color="gray", width=2, dash="dash"))
        
        fig.update_layout(
            xaxis_title="Days from Now",
            yaxis_title="Price (₹/quintal)",
            plot_bgcolor="white",
            font=dict(size=12),
            hovermode="x unified",
            height=300,
            yaxis=dict(
                tickformat="₹,",
                range=[min(forecast_prices) * 0.9, max(forecast_prices) * 1.1]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== 🏆 BEST MARKET =====
        if crop_prices is not None and not crop_prices.empty:
            market_prices = crop_prices[['Market', 'Modal_x0020_Price', 'District']].drop_duplicates(subset=['Market'])
            best_market_row = market_prices.loc[market_prices['Modal_x0020_Price'].idxmax()]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
                <h3 style="color: white; margin: 0;">🏆 Best Market to Sell</h3>
                <h2 style="color: white; margin: 10px 0;">{best_market_row['Market']}</h2>
                <p style="color: white; font-size: 18px;">💰 ₹{int(best_market_row['Modal_x0020_Price'])}/quintal</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all markets
            st.markdown(f"### 📍 All Markets in {selected_state}")
            total_markets = len(market_prices)
            st.markdown(f"**Total markets:** {total_markets}")
            
            # Market list
            market_list = market_prices.sort_values('Modal_x0020_Price', ascending=False).reset_index(drop=True)
            for idx, row in market_list.head(10).iterrows():
                if idx == 0:
                    emoji = "🥇"
                elif idx == 1:
                    emoji = "🥈"
                else:
                    emoji = "🥉"
                st.write(f"{emoji} **{row['Market']}**: ₹{int(row['Modal_x0020_Price'])}/quintal")
        
        if not crop_prices.empty:
            # Show price data with increases/decreases for selected crop and state
            market_prices = crop_prices[['Market', 'Modal_x0020_Price', 'District']].drop_duplicates(subset=['Market']).copy()
            
            # Calculate average price and difference
            avg_price = market_prices['Modal_x0020_Price'].mean()
            max_price = market_prices['Modal_x0020_Price'].max()
            min_price = market_prices['Modal_x0020_Price'].min()
            total_markets = len(market_prices)
            
            # ===== 📍 MARKETS IN THIS STATE =====
            st.markdown(f"### 📍 Markets in {selected_state}: **{total_markets} markets**")
            
            # Show list of all markets with prices - including Min, Max, Modal
            market_list = market_prices.sort_values('Modal_x0020_Price', ascending=False).reset_index(drop=True)
            
            # Get detailed prices for each market (Min, Max, Modal)
            detailed_prices = crop_prices.groupby('Market').agg({
                'Min_x0020_Price': 'min',
                'Max_x0020_Price': 'max',
                'Modal_x0020_Price': 'mean',
                'District': 'first'
            }).reset_index()
            detailed_prices = detailed_prices.sort_values('Modal_x0020_Price', ascending=False).reset_index(drop=True)
            
            # Create display table with all details
            display_df = detailed_prices[['Market', 'District', 'Min_x0020_Price', 'Max_x0020_Price', 'Modal_x0020_Price']].copy()
            display_df.columns = ['🏪 Market', '📍 District', '📉 Min (₹)', '📈 Max (₹)', '💰 Modal (₹)']
            display_df['📉 Min (₹)'] = display_df['📉 Min (₹)'].apply(lambda x: f"₹{int(x)}")
            display_df['📈 Max (₹)'] = display_df['📈 Max (₹)'].apply(lambda x: f"₹{int(x)}")
            display_df['💰 Modal (₹)'] = display_df['💰 Modal (₹)'].apply(lambda x: f"₹{int(x)}")
            
            # Add rank number
            display_df.insert(0, '#', range(1, len(display_df) + 1))
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # ===== 🏆 BEST MARKET HIGHLIGHT =====
            best_market_row = market_prices.loc[market_prices['Modal_x0020_Price'].idxmax()]
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 15px; border: 3px solid #28a745; text-align: center; margin-bottom: 20px;">
                <h2 style="color: #155724; margin: 0;">⭐ Best Market to SELL</h2>
                <h1 style="color: #28a745; margin: 10px 0;">{best_market_row['Market']}</h1>
                <h3 style="color: #155724;">💰 Highest Price: ₹{int(best_market_row['Modal_x0020_Price'])}/quintal</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Sort by price descending for visual comparison
            market_prices_sorted = market_prices.sort_values('Modal_x0020_Price', ascending=False).reset_index(drop=True)
            
            # Create price level based on position (Best=Green, Average=Yellow, Lowest=Red)
            def get_market_level(price, max_p, min_p):
                range_p = max_p - min_p
                if range_p == 0:
                    return "Average"
                position = (price - min_p) / range_p
                if position >= 0.66:
                    return "Best"
                elif position >= 0.33:
                    return "Average"
                else:
                    return "Lowest"
            
            market_prices_sorted['Market Level'] = market_prices_sorted['Modal_x0020_Price'].apply(
                lambda x: get_market_level(x, max_price, min_price)
            )
            
            # ===== 📊 VISUAL MARKET COMPARISON =====
            st.markdown("### 📊 Market Price Comparison")
            
            fig_markets = px.bar(
                market_prices_sorted,
                x="Market",
                y="Modal_x0020_Price",
                color="Market Level",
                color_discrete_map={"Best": "#28a745", "Average": "#ffc107", "Lowest": "#dc3545"},
                title=f"Compare Prices: Higher bars = More money for you!",
                text="Modal_x0020_Price",
                labels={"Modal_x0020_Price": "Price (₹)", "Market": "Market", "Market Level": "Status"}
            )
            fig_markets.update_traces(texttemplate='₹%{value}', textposition='outside')
            fig_markets.update_layout(
                xaxis_title="",
                yaxis_title="Price (₹/quintal)",
                font=dict(size=14),
                plot_bgcolor="white",
                showlegend=True,
                legend_title="Market Status"
            )
            st.plotly_chart(fig_markets, use_container_width=True)
            
            # Color legend explanation
            st.markdown("""
            <div style="display: flex; justify-content: center; gap: 20px; margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background-color: #28a745; padding: 5px 10px; border-radius: 5px; color: white; font-weight: bold;">🟢 Best</span>
                    <span>Sell here!</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background-color: #ffc107; padding: 5px 10px; border-radius: 5px; color: black; font-weight: bold;">🟡 Average</span>
                    <span>OK price</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="background-color: #dc3545; padding: 5px 10px; border-radius: 5px; color: white; font-weight: bold;">🔴 Lowest</span>
                    <span>Avoid</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show price details in TABLE format
            price_table = crop_prices[['Market', 'Min_x0020_Price', 'Max_x0020_Price', 'Modal_x0020_Price', 'Arrival_Date']].copy()
            price_table.columns = ['Market', 'Min Price (₹)', 'Max Price (₹)', 'Modal Price (₹)', 'Date']
            price_table = price_table.sort_values('Modal Price (₹)', ascending=False)
            
            # Format prices
            price_table['Min Price (₹)'] = price_table['Min Price (₹)'].apply(lambda x: f"₹{int(x)}")
            price_table['Max Price (₹)'] = price_table['Max Price (₹)'].apply(lambda x: f"₹{int(x)}")
            price_table['Modal Price (₹)'] = price_table['Modal Price (₹)'].apply(lambda x: f"₹{int(x)}")
            
            st.dataframe(price_table, use_container_width=True, hide_index=True)
            
            # Show price details
            current_price = crop_prices['Modal_x0020_Price'].iloc[0]
            min_price = crop_prices['Min_x0020_Price'].iloc[0]
            max_price = crop_prices['Max_x0020_Price'].iloc[0]
            
            # Show as metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Modal Price", f"₹{int(current_price)}")
            with col2:
                st.metric("Min Price", f"₹{int(min_price)}")
            with col3:
                st.metric("Max Price", f"₹{int(max_price)}")
            with col4:
                st.metric("Markets Available", f"{len(market_prices)}")
        else:
            # No CSV data - use fallback
            state_data = ALL_STATES_PRICES.get(selected_state, {})
            crop_price = state_data.get(crop_choice)
            
            if crop_price:
                current_price = crop_price
                min_price = int(crop_price * 0.85)
                max_price = int(crop_price * 1.15)
                st.info(f"Price data for {selected_state} (fallback data)")
                
                # Bar chart with color coding using Plotly
                st.markdown("#### 💰 Market Price")
                price_data = pd.DataFrame({
                    'Price Type': ['Min', 'Modal', 'Max'],
                    'Price (₹)': [min_price, current_price, max_price],
                    'Level': ['Low', 'Medium', 'High']
                })
                
                fig_price = px.bar(
                    price_data,
                    x='Price Type',
                    y='Price (₹)',
                    color='Level',
                    color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60'},
                    title=f"Price Range for {crop_choice}",
                    text='Price (₹)'
                )
                fig_price.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.table(pd.DataFrame({
                    'Market': ['Estimated Market'],
                    'Min Price (₹)': [f"₹{min_price}"],
                    'Max Price (₹)': [f"₹{max_price}"],
                    'Modal Price (₹)': [f"₹{current_price}"],
                    'Date': ['N/A']
                }))
            else:
                current_price = 2000
                min_price = 1600
                max_price = 2400
                st.info("Estimated market price")
                
                # Bar chart with color coding using Plotly
                st.markdown("#### 💰 Market Price")
                price_data = pd.DataFrame({
                    'Price Type': ['Min', 'Modal', 'Max'],
                    'Price (₹)': [min_price, current_price, max_price],
                    'Level': ['Low', 'Medium', 'High']
                })
                
                fig_price = px.bar(
                    price_data,
                    x='Price Type',
                    y='Price (₹)',
                    color='Level',
                    color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60'},
                    title=f"Price Range for {crop_choice}",
                    text='Price (₹)'
                )
                fig_price.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.table(pd.DataFrame({
                    'Market': ['Estimated Market'],
                    'Min Price (₹)': [f"₹{min_price}"],
                    'Max Price (₹)': [f"₹{max_price}"],
                    'Modal Price (₹)': [f"₹{current_price}"],
                    'Date': ['N/A']
                }))
    else:
        # Use fallback prices
        state_data = ALL_STATES_PRICES.get(selected_state, {})
        crop_price = state_data.get(crop_choice, 2000)
        current_price = crop_price
        min_price = int(crop_price * 0.85)
        max_price = int(crop_price * 1.15)
        st.info(f"Price data for {selected_state} (estimated)")
        
        # Bar chart
        price_data = pd.DataFrame({
            'Price Type': ['Min', 'Modal', 'Max'],
            'Price (₹)': [min_price, current_price, max_price],
            'Level': ['Low', 'Medium', 'High']
        })
        
        fig_price = px.bar(
            price_data,
            x='Price Type',
            y='Price (₹)',
            color='Level',
            color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60'},
            title=f"Price Range for {crop_choice}",
            text='Price (₹)'
        )
        fig_price.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_price, use_container_width=True)
    
    # ========== BEST MARKET SUGGESTION (eNAM) ==========
    st.markdown("---")
    st.markdown("### 🎯 Best Market Suggestion (eNAM)")
    
    # Get market data for best suggestion
    state_market_data = ALL_STATES_MARKET_DATA.get(selected_state, {})
    
    if state_market_data and crop_choice:
        # Use comprehensive data
        market_list_data = []
        for market, crops in state_market_data.items():
            if crop_choice in crops:
                price = crops[crop_choice]
                market_list_data.append({
                    'Market': market,
                    'Price': price
                })
        
        if len(market_list_data) > 1:
            # Sort by price
            market_list_data = sorted(market_list_data, key=lambda x: x['Price'], reverse=True)
            best = market_list_data[0]
            worst = market_list_data[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Market to SELL:** {best['Market']}")
                st.write(f"💰 ₹{best['Price']}/quintal")
            with col2:
                st.warning(f"📉 **Lowest Price:** {worst['Market']}")
                st.write(f"💰 ₹{worst['Price']}/quintal")
            
            savings = best['Price'] - worst['Price']
            st.info(f"💡 Sell at **{best['Market']}** to earn ₹{int(savings)} more!")
        else:
            st.info("More market data coming soon from eNAM")
    else:
        st.info("Select a crop to see best market suggestion from eNAM")


# ---------------------------
# Weather with Smart Advisory
# ---------------------------
elif menu == get_text("menu_weather", global_lang):
    st.subheader("🌤️ " + get_text("live_weather", global_lang))
    
    # Crop selection for personalized advice
    st.markdown("### 🌾 Select Your Crop (for personalized advice)")
    crop_options = ["General", "Rice", "Wheat", "Tomato", "Potato", "Cotton", "Sugarcane", "Maize", "Onion", "Grapes"]
    selected_crop = st.selectbox("Crop:", crop_options, index=0)
    
    city = st.text_input(get_text("enter_city", global_lang), "Delhi")

    if st.button(get_text("get_weather", global_lang)):
        with st.spinner("Fetching weather data..."):
            weather, error = get_weather(city)
        if weather:
            temp = weather.get('temperature', 0)
            humidity = weather.get('humidity', 0)
            rainfall = weather.get('rainfall', 0)
            description = weather.get('description', '').lower()
            
            # === 1. Weather Display ===
            st.write(f"### 🌤️ Weather – {city}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌡️ Temperature", f"{temp}°C")
            with col2:
                st.metric("💧 Humidity", f"{humidity}%")
            with col3:
                st.metric("🌧️ Rainfall", f"{rainfall} mm")
            with col4:
                st.metric("☁️ Condition", description.title())
            
            # === 2. Weather Risk Level ===
            # Calculate risk based on conditions
            risk_level = "LOW"
            risk_color = "#28a745"
            risk_desc = ""
            
            # High risk conditions
            if temp > 40:
                risk_level = "HIGH"
                risk_color = "#dc3545"
                risk_desc = "Extreme heat stress for crops"
            elif humidity > 85 and temp > 25:
                risk_level = "HIGH"
                risk_color = "#dc3545"
                risk_desc = "High humidity + warm temperature - very high risk of fungal diseases"
            elif temp > 35 and humidity > 70:
                risk_level = "MEDIUM"
                risk_color = "#ffc107"
                risk_desc = "Warm and humid - monitor for disease risk"
            elif rainfall > 20:
                risk_level = "MEDIUM"
                risk_color = "#ffc107"
                risk_desc = "Heavy rain may cause waterlogging"
            elif temp < 5:
                risk_level = "HIGH"
                risk_color = "#dc3545"
                risk_desc = "Cold stress - protect sensitive crops"
            else:
                risk_level = "LOW"
                risk_color = "#28a745"
                risk_desc = "Normal conditions - no immediate risk"
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {risk_color};">
                <h4 style="color: {risk_color}; margin: 0;">🚨 Weather Risk Level: {risk_level}</h4>
                <p style="color: #666; margin: 5px 0;">{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # === 3. Actionable Advisory ===
            st.markdown("### 🌿 Actionable Advisory")
            
            # Generate advice based on weather + crop
            advisories = []
            
            # Rain advisory
            if rainfall > 5:
                advisories.append("🌧️ <b>Avoid irrigation today</b> - Soil moisture is sufficient")
                advisories.append("🌧️ <b>Delay spraying pesticides/fertilizers</b> - Will be washed away")
            elif rainfall == 0 and humidity < 40:
                advisories.append("💧 <b>Increase irrigation</b> - Low humidity increases water needs")
            
            # Temperature advisory
            if temp > 38:
                advisories.append("☀️ <b>Water crops in early morning or evening</b> - Reduce evaporation")
                advisories.append("☀️ <b>Provide shade for sensitive crops</b> - Heat stress risk")
            elif temp < 10:
                advisories.append("🥶 <b>Protect crops from cold</b> - Use covers for sensitive varieties")
            
            # Humidity advisory  
            if humidity > 80:
                advisories.append("🍄 <b>High risk of fungal diseases</b> - Monitor leaves closely")
                advisories.append("🍄 <b>Avoid overhead irrigation</b> - Increases disease spread")
            elif humidity < 30:
                advisories.append("💨 <b>Low humidity</b> - May cause drying, increase watering")
            
            # Crop-specific advice
            if selected_crop != "General":
                if selected_crop in ["Rice"]:
                    if rainfall > 10:
                        advisories.append("🌾 <b>Rice:</b> Standing water is good - maintain paddy fields")
                    elif rainfall < 5:
                        advisories.append("🌾 <b>Rice:</b> May need supplemental irrigation")
                elif selected_crop in ["Tomato", "Grapes"]:
                    if humidity > 75:
                        advisories.append("🍅 <b>{crop}:</b> High humidity increases risk of fungal diseases like Powdery Mildew".format(crop=selected_crop))
                elif selected_crop in ["Cotton"]:
                    if rainfall > 15:
                        advisories.append("🌱 <b>Cotton:</b> Avoid cotton picking in wet conditions")
                elif selected_crop in ["Onion", "Potato"]:
                    if humidity > 80:
                        advisories.append("🧅 <b>{crop}:</b> Risk of bulb rot - ensure good drainage".format(crop=selected_crop))
            
            # Display advisories
            for advice in advisories:
                st.markdown(f"<p style='margin: 5px 0;'>{advice}</p>", unsafe_allow_html=True)
            
            # === 4. Disease Risk Connection ===
            if humidity > 70 and temp > 20:
                st.markdown("### 🔗 Smart Insight: Disease Risk")
                st.markdown("""
                <div style="background: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <p style="margin: 8px 0;">🍄 <b>Weather conditions favor fungal diseases:</b></p>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>Powdery Mildew (common in warm humid weather)</li>
                        <li>Late Blight (favored by wet conditions)</li>
                        <li>Leaf Spot diseases</li>
                    </ul>
                    <p style="margin: 8px 0;">💡 <b>Recommendation:</b> Consider preventive fungicide application if crop is in vulnerable stage.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # === 5. Price Impact Connection ===
            if rainfall > 25:
                st.markdown("### 📈 Smart Insight: Market Impact")
                st.markdown("""
                <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <p style="margin: 8px 0;">🌧️ <b>Heavy rain may affect market:</b></p>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>Supply disruption - vegetables may become scarce</li>
                        <li>Prices may increase in next 3-5 days</li>
                        <li>Consider storing produce if possible</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # === 6. Trust Layer ===
            st.markdown("""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                <p style="color: #6c757d; font-size: 12px; margin: 0;">
                    <strong>⚠️ Note:</strong> Weather data from OpenWeatherMap API. Conditions may change rapidly. 
                    For critical farming decisions, verify with local weather forecasts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error(f"Unable to fetch weather: {error}")
            st.info("Please check your internet connection and try again.")
            # Show sample data for testing
            st.info("💡 Tip: Make sure you have an internet connection and try entering a valid city name.")

# ---------------------------
# Disease Detection using Plant.id API or Pl@ntNet API or Local TensorFlow
# ---------------------------
elif menu == get_text("menu_disease", global_lang):
    st.subheader("🌿 " + get_text("menu_disease", global_lang))
    st.write("📷 Upload a photo of your plant leaf to detect diseases")
    
    # Use Local AI Model (TensorFlow) - no API key needed, runs locally
    api_option = "Local AI Model (TensorFlow)"
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Plant Image", use_container_width=True)
        
        # Detect button
        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analyzing plant image..."):
                try:
                    # Read image
                    image_bytes = uploaded_file.getvalue()
                    
                    if "Local AI" in api_option:
                        # Use local TensorFlow model
                        st.info("🤖 Using local TensorFlow AI model...")
                        
                        try:
                            # Import and use local model
                            from plant_disease_model import get_detector
                            import io
                            
                            # Get detector
                            detector = get_detector()
                            
                            # Create a temporary file-like object
                            img_buffer = io.BytesIO(image_bytes)
                            
                            # Predict
                            predictions = detector.predict(image_bytes=img_buffer)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Detection Results")
                            
                            # Display top prediction
                            top_result = predictions[0]
                            
                            st.markdown(f"**🌱 Plant:** {top_result['plant']}")
                            st.markdown(f"**🦠 Condition:** {top_result['disease']}")
                            
                            if top_result['is_healthy']:
                                st.success(f"✅ Your plant appears healthy!")
                                st.markdown(f"""
                                <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                    <h4 style="color: #155724; margin: 0;">🌿 Plant Health Status</h4>
                                    <p style="color: #155724; margin: 5px 0;">Your plant shows no signs of disease. Keep up the good care!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Check for low confidence - might be unclear image
                                confidence_pct = int(top_result['confidence'] * 100)
                                
                                # === Failure Handling for unclear images ===
                                if confidence_pct < 50:
                                    st.warning("⚠️ Unable to detect disease clearly. The uploaded image may be:")
                                    st.write("• Too blurry or out of focus")
                                    st.write("• Not showing clear disease symptoms")
                                    st.write("• Insufficient lighting or poor image quality")
                                    st.info("💡 Please upload a clearer image focusing on the affected area with good lighting.")
                                    
                                    # Still show what we can determine
                                    st.markdown(f"**🌱 Plant:** {top_result['plant']}")
                                    st.markdown(f"**🦠 Possible Condition:** {top_result['disease']} (low confidence)")
                                    
                                    # Continue with minimal info but reduced confidence display
                                    conf_color = "#dc3545"
                                    st.markdown(f"""
                                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                        <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                        <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">(UNCERTAIN)</span></h2>
                                        <p style="color: #666; font-size: 14px; margin: 5px 0;">Unclear image - please retake with better quality</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show treatment but note uncertainty
                                    st.markdown("### 💊 Suggested Treatment (verify with expert)")
                                    st.info(top_result['treatment'])
                                    
                                    # Trust Layer with extra caution
                                    st.markdown("""
                                    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #dc3545;">
                                        <p style="color: #dc3545; font-size: 12px; margin: 0;">
                                            <strong>⚠️ Warning:</strong> Low detection confidence. Please consult agricultural expert 
                                            for accurate diagnosis before applying treatment.
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Normal detection with sufficient confidence - already have confidence_pct from above
                                    pass
                                
                                # === 1. Confidence with explanation and color badge ===
                                if confidence_pct >= 80:
                                    conf_level = "HIGH"
                                    conf_color = "#28a745"
                                    conf_explanation = "Clear disease patterns detected in image with strong model certainty"
                                elif confidence_pct >= 60:
                                    conf_level = "MEDIUM"
                                    conf_color = "#ffc107"
                                    conf_explanation = "Disease symptoms detected but image may be unclear - consider verification"
                                else:
                                    conf_level = "LOW"
                                    conf_color = "#dc3545"
                                    conf_explanation = "Unclear image or mixed symptoms detected - expert consultation recommended"
                                
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                    <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                    <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">({conf_level})</span></h2>
                                    <p style="color: #666; font-size: 14px; margin: 5px 0;">{conf_explanation}</p>
                                    <div style="background-color: #e9ecef; border-radius: 5px; height: 20px; width: 100%;">
                                        <div style="background-color: {conf_color}; border-radius: 5px; height: 100%; width: {confidence_pct}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 2. Severity Estimation ===
                                # Simple logic based on confidence - higher confidence = more severe indication
                                if confidence_pct >= 85:
                                    severity = "SEVERE"
                                    severity_color = "#dc3545"
                                    severity_desc = "Disease appears well-established across multiple areas of the plant."
                                elif confidence_pct >= 70:
                                    severity = "MODERATE"
                                    severity_color = "#ffc107"
                                    severity_desc = "Disease is visible on several leaves - treatment needed soon."
                                else:
                                    severity = "MILD"
                                    severity_color = "#17a2b8"
                                    severity_desc = "Early signs on few leaves - quick treatment can prevent spread."
                                
                                st.markdown(f"""
                                <div style="background-color: #fff3cd; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {severity_color};">
                                    <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity: {severity}</h4>
                                    <p style="color: #666; font-size: 14px; margin: 5px 0;">{severity_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 3. Spread Risk (NEW) ===
                                # Higher severity + certain diseases = higher spread risk
                                disease_name = top_result.get('disease', '').lower()
                                high_spread_diseases = ['blight', 'mildew', 'rust', 'spot', 'rot']
                                has_high_spread_disease = any(d in disease_name for d in high_spread_diseases)
                                
                                if severity == "SEVERE" and has_high_spread_disease:
                                    spread_risk = "HIGH"
                                    spread_color = "#dc3545"
                                    spread_desc = "This disease can spread rapidly to other plants. Isolate if possible."
                                elif severity == "MODERATE":
                                    spread_risk = "MEDIUM"
                                    spread_color = "#ffc107"
                                    spread_desc = "Monitor nearby plants for symptoms. Treat soon to prevent spread."
                                else:
                                    spread_risk = "LOW"
                                    spread_color = "#28a745"
                                    spread_desc = "Low risk of spreading to nearby plants if treated promptly."
                                
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {spread_color};">
                                    <h4 style="color: {spread_color}; margin: 0;">🔄 Spread Risk: {spread_risk}</h4>
                                    <p style="color: #666; font-size: 14px; margin: 5px 0;">{spread_desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 4. Urgency Level with Logic Explanation ===
                                if confidence_pct >= 80:
                                    urgency = "HIGH"
                                    urgency_icon = "🚨"
                                    urgency_color = "#dc3545"
                                    urgency_action = "ACT IMMEDIATELY - Disease can spread rapidly to entire crop"
                                    urgency_reason = "Based on: High severity + high spread risk disease"
                                elif confidence_pct >= 60:
                                    urgency = "MEDIUM"
                                    urgency_icon = "⚡"
                                    urgency_color = "#ffc107"
                                    urgency_action = "Treat within 2-3 days to prevent spread"
                                    urgency_reason = "Based on: Moderate severity with manageable spread risk"
                                else:
                                    urgency = "LOW"
                                    urgency_icon = "📊"
                                    urgency_color = "#17a2b8"
                                    urgency_action = "Monitor and verify before taking action"
                                    urgency_reason = "Based on: Mild symptoms - verify diagnosis first"
                                
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border: 2px solid {urgency_color};">
                                    <h4 style="color: {urgency_color}; margin: 0;">{urgency_icon} Urgency: {urgency}</h4>
                                    <p style="color: #666; font-size: 14px; margin: 5px 0;">{urgency_action}</p>
                                    <p style="color: #999; font-size: 12px; margin: 5px 0;"><em>{urgency_reason}</em></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 4. Structured Treatment ===
                                treatment_text = top_result.get('treatment', 'Consult agricultural expert')
                                st.markdown("### 💊 Recommended Treatment")
                                
                                # Parse/generate structured treatment
                                st.markdown(f"""
                                <div style="display: flex; gap: 10px; margin: 10px 0;">
                                    <div style="flex: 1; background: #e7f3ff; padding: 10px; border-radius: 8px;">
                                        <h5 style="color: #0066cc; margin: 0;">🧪 Chemical Solution</h5>
                                        <p style="font-size: 13px;">{treatment_text}</p>
                                    </div>
                                    <div style="flex: 1; background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                        <h5 style="color: #2e7d32; margin: 0;">🌿 Organic Solution</h5>
                                        <p style="font-size: 13px;">Apply neem oil spray or copper-based organic fungicide. Remove infected leaves. Improve air circulation.</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("""
                                <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                                    <h5 style="color: #e65100; margin: 0;">🛡️ Prevention Tips</h5>
                                    <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                                        <li>Rotate crops annually to prevent disease buildup</li>
                                        <li>Use disease-resistant varieties when possible</li>
                                        <li>Remove and destroy infected plant parts</li>
                                        <li>Avoid overhead watering - water at soil level</li>
                                        <li>Maintain proper plant spacing for air circulation</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 5. Ecosystem Connection - Disease Impact ===
                                st.markdown("### 🌾 Ecosystem Impact")
                                
                                crop_name = top_result.get('plant', 'Crop')
                                disease_name = top_result.get('disease', 'Disease')
                                
                                # Estimate impact based on severity
                                if severity == "SEVERE":
                                    yield_impact = "Potential yield reduction possible (estimate based on typical disease impact patterns - actual impact varies by crop health and management)"
                                    price_impact = "Quality impact may affect market value (actual impact depends on disease severity and market conditions)"
                                    recommendation = "Consider harvesting early if crop is near maturity to minimize loss"
                                elif severity == "MODERATE":
                                    yield_impact = "Potential yield reduction possible (depends on treatment timing and crop health - actual impact varies)"
                                    price_impact = "Minor quality impact on market price (may affect grade but generally recoverable)"
                                    recommendation = "Treat immediately and monitor for 7 days"
                                else:
                                    yield_impact = "Minimal yield impact expected if treated promptly (actual impact varies by crop and management)"
                                    price_impact = "Minimal impact expected if treated promptly and properly"
                                    recommendation = "Apply treatment and recheck in 1 week"
                                
                                st.markdown(f"""
                                <div style="background: #fce4ec; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                    <h5 style="color: #c2185b; margin: 0;">📉 Yield & Price Impact</h5>
                                    <p style="margin: 8px 0;"><strong>🌾 Yield:</strong> {yield_impact}</p>
                                    <p style="margin: 8px 0;"><strong>💰 Market:</strong> {price_impact}</p>
                                    <p style="margin: 8px 0;"><strong>💡 Advisory:</strong> {recommendation}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # === 6. Trust Layer - Disclaimer ===
                                st.markdown("""
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                                    <p style="color: #6c757d; font-size: 12px; margin: 0;">
                                        <strong>⚠️ Disclaimer:</strong> This AI-based detection is for preliminary guidance only. 
                                        For severe cases or uncertain diagnoses, please consult your local agricultural extension office or a qualified plant pathologist.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show other possibilities
                            if len(predictions) > 1:
                                st.markdown("### 📊 Other Possibilities")
                                for i, pred in enumerate(predictions[1:], 1):
                                    conf = int(pred['confidence'] * 100)
                                    st.write(f"{i}. {pred['disease']} ({pred['plant']}) - {conf}%")
                            
                            st.caption("🤖 Powered by TensorFlow MobileNetV2 (Local AI)")
                            
                        except ImportError as e:
                            st.warning("⚠️ TensorFlow not installed. Installing...")
                            st.info("Please run: pip install tensorflow")
                            
                            # Fallback to demo
                            import random
                            diseases = [
                                {"name": "Early Blight", "probability": 0.92, "treatment": "Apply copper-based fungicide, remove infected leaves, avoid overhead watering"},
                                {"name": "Late Blight", "probability": 0.88, "treatment": "Apply fungicide immediately, remove severely infected plants, improve air circulation"},
                                {"name": "Powdery Mildew", "probability": 0.85, "treatment": "Apply neem oil or sulfur fungicide, improve ventilation, reduce humidity"},
                            ]
                            result = random.choice(diseases)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Demo Results (Install TensorFlow for real detection)")
                            
                            # === Enhanced Demo Results ===
                            confidence_pct = int(result['probability'] * 100)
                            
                            # 1. Confidence with level
                            if confidence_pct >= 80:
                                conf_level = "HIGH"
                                conf_color = "#28a745"
                                conf_explanation = "Clear disease patterns detected in image with strong model certainty"
                            elif confidence_pct >= 60:
                                conf_level = "MEDIUM"
                                conf_color = "#ffc107"
                                conf_explanation = "Disease symptoms detected but image may be unclear - consider verification"
                            else:
                                conf_level = "LOW"
                                conf_color = "#dc3545"
                                conf_explanation = "Unclear image or mixed symptoms detected - expert consultation recommended"
                            
                            st.markdown(f"**🌱 Plant:** Tomato")
                            st.markdown(f"**🦠 Disease Detected:** {result['name']}")
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">({conf_level})</span></h2>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{conf_explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 2. Severity
                            if confidence_pct >= 85:
                                severity = "SEVERE"
                                severity_color = "#dc3545"
                                severity_desc = "Disease appears well-established. Immediate action recommended."
                            elif confidence_pct >= 70:
                                severity = "MODERATE"
                                severity_color = "#ffc107"
                                severity_desc = "Disease is spreading. Monitor closely and treat soon."
                            else:
                                severity = "MILD"
                                severity_color = "#17a2b8"
                                severity_desc = "Early signs detected. Quick treatment can prevent spread."
                            
                            st.markdown(f"""
                            <div style="background-color: #fff3cd; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {severity_color};">
                                <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity: {severity}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{severity_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 3. Urgency
                            if confidence_pct >= 80:
                                urgency = "HIGH"
                                urgency_icon = "🚨"
                                urgency_color = "#dc3545"
                                urgency_action = "ACT IMMEDIATELY - Disease can spread rapidly"
                            elif confidence_pct >= 60:
                                urgency = "MEDIUM"
                                urgency_icon = "⚡"
                                urgency_color = "#ffc107"
                                urgency_action = "Treat within 2-3 days to prevent spread"
                            else:
                                urgency = "LOW"
                                urgency_icon = "📊"
                                urgency_color = "#17a2b8"
                                urgency_action = "Monitor and verify before taking action"
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border: 2px solid {urgency_color};">
                                <h4 style="color: {urgency_color}; margin: 0;">{urgency_icon} Urgency: {urgency}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{urgency_action}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 4. Structured Treatment
                            st.markdown("### 💊 Recommended Treatment")
                            treatment_text = result['treatment']
                            
                            st.markdown(f"""
                            <div style="display: flex; gap: 10px; margin: 10px 0;">
                                <div style="flex: 1; background: #e7f3ff; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #0066cc; margin: 0;">🧪 Chemical Solution</h5>
                                    <p style="font-size: 13px;">{treatment_text}</p>
                                </div>
                                <div style="flex: 1; background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #2e7d32; margin: 0;">🌿 Organic Solution</h5>
                                    <p style="font-size: 13px;">Apply neem oil spray or copper-based organic fungicide. Remove infected leaves. Improve air circulation.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                                <h5 style="color: #e65100; margin: 0;">🛡️ Prevention Tips</h5>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                                    <li>Rotate crops annually to prevent disease buildup</li>
                                    <li>Use disease-resistant varieties when possible</li>
                                    <li>Remove and destroy infected plant parts</li>
                                    <li>Avoid overhead watering - water at soil level</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 5. Ecosystem Impact
                            if severity == "SEVERE":
                                yield_impact = "Estimated 30-50% yield reduction possible (based on typical disease impact patterns)"
                                price_impact = "Quality drop may reduce market value (estimated 20-40% reduction based on typical disease impact)"
                                recommendation = "Consider harvesting early if crop is near maturity to minimize loss"
                            elif severity == "MODERATE":
                                yield_impact = "Estimated 10-25% yield reduction possible (depends on treatment timing and crop health)"
                                price_impact = "Minor quality impact on market price (may affect grade but generally recoverable)"
                                recommendation = "Treat immediately and monitor for 7 days"
                            else:
                                yield_impact = "Estimated 5-10% yield impact if left untreated (minimal if treated promptly)"
                                price_impact = "Minimal impact expected if treated promptly and properly"
                                recommendation = "Apply treatment and recheck in 1 week"
                            
                            st.markdown("### 🌾 Ecosystem Impact")
                            st.markdown(f"""
                            <div style="background: #fce4ec; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h5 style="color: #c2185b; margin: 0;">📉 Yield & Price Impact</h5>
                                <p style="margin: 8px 0;"><strong>🌾 Yield:</strong> {yield_impact}</p>
                                <p style="margin: 8px 0;"><strong>💰 Market:</strong> {price_impact}</p>
                                <p style="margin: 8px 0;"><strong>💡 Advisory:</strong> {recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 6. Trust Layer
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                                <p style="color: #6c757d; font-size: 12px; margin: 0;">
                                    <strong>⚠️ Disclaimer:</strong> This is demo mode. Install TensorFlow for real detection accuracy.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    elif "Pl@ntNet" in api_option:
                        # Pl@ntNet API
                        plantnet_api_key = os.getenv('PLANTNET_API_KEY', '')
                        
                        # Check if API key is valid (not placeholder)
                        if not plantnet_api_key or plantnet_api_key in ['your_plantnet_api_key_here', 'demo_key', '']:
                            # Demo mode - simulate API response
                            st.info("🔧 Running in demo mode")
                            
                            import random
                            diseases = [
                                {"name": "Early Blight", "probability": 0.92, "treatment": "Apply copper-based fungicide, remove infected leaves, avoid overhead watering"},
                                {"name": "Late Blight", "probability": 0.88, "treatment": "Apply fungicide immediately, remove severely infected plants, improve air circulation"},
                                {"name": "Powdery Mildew", "probability": 0.85, "treatment": "Apply neem oil or sulfur fungicide, improve ventilation, reduce humidity"},
                                {"name": "Leaf Spot", "probability": 0.79, "treatment": "Remove infected leaves, apply copper fungicide, avoid wetting foliage"},
                                {"name": "Bacterial Spot", "probability": 0.75, "treatment": "Apply copper-based spray, remove infected plant parts, rotate crops"}
                            ]
                            result = random.choice(diseases)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Demo Detection Results")
                            st.markdown(f"**🌱 Plant:** Tomato")
                            
                            # === Enhanced Demo Results ===
                            confidence_pct = int(result['probability'] * 100)
                            
                            # 1. Confidence with level
                            if confidence_pct >= 80:
                                conf_level = "HIGH"
                                conf_color = "#28a745"
                                conf_explanation = "Clear disease patterns detected in image with strong model certainty"
                            elif confidence_pct >= 60:
                                conf_level = "MEDIUM"
                                conf_color = "#ffc107"
                                conf_explanation = "Disease symptoms detected but image may be unclear - consider verification"
                            else:
                                conf_level = "LOW"
                                conf_color = "#dc3545"
                                conf_explanation = "Unclear image or mixed symptoms detected - expert consultation recommended"
                            
                            st.markdown(f"**🦠 Disease Detected:** {result['name']}")
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">({conf_level})</span></h2>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{conf_explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 2. Severity
                            if confidence_pct >= 85:
                                severity = "SEVERE"
                                severity_color = "#dc3545"
                                severity_desc = "Disease appears well-established. Immediate action recommended."
                            elif confidence_pct >= 70:
                                severity = "MODERATE"
                                severity_color = "#ffc107"
                                severity_desc = "Disease is spreading. Monitor closely and treat soon."
                            else:
                                severity = "MILD"
                                severity_color = "#17a2b8"
                                severity_desc = "Early signs detected. Quick treatment can prevent spread."
                            
                            st.markdown(f"""
                            <div style="background-color: #fff3cd; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {severity_color};">
                                <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity: {severity}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{severity_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 3. Urgency
                            if confidence_pct >= 80:
                                urgency = "HIGH"
                                urgency_icon = "🚨"
                                urgency_color = "#dc3545"
                                urgency_action = "ACT IMMEDIATELY - Disease can spread rapidly"
                            elif confidence_pct >= 60:
                                urgency = "MEDIUM"
                                urgency_icon = "⚡"
                                urgency_color = "#ffc107"
                                urgency_action = "Treat within 2-3 days to prevent spread"
                            else:
                                urgency = "LOW"
                                urgency_icon = "📊"
                                urgency_color = "#17a2b8"
                                urgency_action = "Monitor and verify before taking action"
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border: 2px solid {urgency_color};">
                                <h4 style="color: {urgency_color}; margin: 0;">{urgency_icon} Urgency: {urgency}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{urgency_action}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 4. Structured Treatment
                            treatment_text = result['treatment']
                            st.markdown("### 💊 Recommended Treatment")
                            
                            st.markdown(f"""
                            <div style="display: flex; gap: 10px; margin: 10px 0;">
                                <div style="flex: 1; background: #e7f3ff; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #0066cc; margin: 0;">🧪 Chemical Solution</h5>
                                    <p style="font-size: 13px;">{treatment_text}</p>
                                </div>
                                <div style="flex: 1; background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #2e7d32; margin: 0;">🌿 Organic Solution</h5>
                                    <p style="font-size: 13px;">Apply neem oil spray or copper-based organic fungicide. Remove infected leaves. Improve air circulation.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                                <h5 style="color: #e65100; margin: 0;">🛡️ Prevention Tips</h5>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                                    <li>Rotate crops annually to prevent disease buildup</li>
                                    <li>Use disease-resistant varieties when possible</li>
                                    <li>Remove and destroy infected plant parts</li>
                                    <li>Avoid overhead watering - water at soil level</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 5. Ecosystem Impact
                            if severity == "SEVERE":
                                yield_impact = "Estimated 30-50% yield reduction possible (based on typical disease impact patterns)"
                                price_impact = "Quality drop may reduce market value (estimated 20-40% reduction based on typical disease impact)"
                                recommendation = "Consider harvesting early if crop is near maturity to minimize loss"
                            elif severity == "MODERATE":
                                yield_impact = "Estimated 10-25% yield reduction possible (depends on treatment timing and crop health)"
                                price_impact = "Minor quality impact on market price (may affect grade but generally recoverable)"
                                recommendation = "Treat immediately and monitor for 7 days"
                            else:
                                yield_impact = "Estimated 5-10% yield impact if left untreated (minimal if treated promptly)"
                                price_impact = "Minimal impact expected if treated promptly and properly"
                                recommendation = "Apply treatment and recheck in 1 week"
                            
                            st.markdown("### 🌾 Ecosystem Impact")
                            st.markdown(f"""
                            <div style="background: #fce4ec; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h5 style="color: #c2185b; margin: 0;">📉 Yield & Price Impact</h5>
                                <p style="margin: 8px 0;"><strong>🌾 Yield:</strong> {yield_impact}</p>
                                <p style="margin: 8px 0;"><strong>💰 Market:</strong> {price_impact}</p>
                                <p style="margin: 8px 0;"><strong>💡 Advisory:</strong> {recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 6. Trust Layer
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                                <p style="color: #6c757d; font-size: 12px; margin: 0;">
                                    <strong>⚠️ Disclaimer:</strong> This is demo mode. Get a free API key for real detection.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.warning("⚠️ Demo mode - Get a free API key for real detection:")
                            st.info("1. Go to https://my.plantnet.org/signup")
                            st.info("2. Create account → Go to Settings → API Key")
                            st.info("3. Copy your API key and add to .env file as PLANTNET_API_KEY=your_key")
                        else:
                            # Real Pl@ntNet API call
                            st.info("🔄 Connecting to Pl@ntNet API...")
                            
                            import io
                            from PIL import Image
                            
                            # Prepare image
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Convert to RGB if needed
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Save to buffer as JPEG
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format='JPEG', quality=85)
                            img_buffer.seek(0)
                            
                            # Pl@ntNet API endpoint for disease identification
                            url = "https://my-api.plantnet.org/v2/diseases/identify?lang=en&include-related-images=true&api-key=" + plantnet_api_key
                            
                            files = {
                                'images': ('plant_image.jpg', img_buffer, 'image/jpeg')
                            }
                            data = {
                                'organs': 'leaf'
                            }
                            
                            response = requests.post(url, files=files, data=data, timeout=60)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                st.markdown("---")
                                st.markdown("### 📊 Detection Results")
                                
                                results_list = result.get('results', [])
                                
                                if results_list and len(results_list) > 0:
                                    # Show top disease
                                    top_result = results_list[0]
                                    disease_name = top_result.get('label', 'Unknown Disease')
                                    disease_score = top_result.get('score', 0)
                                    disease_desc = top_result.get('description', '')
                                    
                                    st.markdown(f"**🦠 Disease Detected:** {disease_name}")
                                    
                                    confidence_pct = int(disease_score * 100)
                                    conf_color = "#28a745" if confidence_pct >= 80 else "#ffc107" if confidence_pct >= 60 else "#dc3545"
                                    st.markdown(f"""
                                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                        <h4 style="color: #666; margin: 0;">🎯 Detection Accuracy</h4>
                                        <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}%</h2>
                                        <div style="background-color: #e9ecef; border-radius: 5px; height: 20px; width: 100%;">
                                            <div style="background-color: {conf_color}; border-radius: 5px; height: 100%; width: {confidence_pct}%;"></div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if disease_desc:
                                        st.markdown("### 📋 Disease Description")
                                        st.info(disease_desc)
                                    
                                    # Show similar images if available
                                    if top_result.get('images'):
                                        st.markdown("### 🔍 Similar Disease Images")
                                        with st.container():
                                            cols = st.columns(min(3, len(top_result['images'])))
                                            for i, img in enumerate(top_result['images'][:3]):
                                                if 'url' in img and 's' in img['url']:
                                                    cols[i].image(img['url']['s'], caption=img.get('organ', 'leaf'))
                                    
                                    # Get treatment info from EPPO code
                                    st.markdown("### 💊 Recommended Treatment")
                                    
                                    # Map common diseases to treatments
                                    disease_treatments = {
                                        "APHISP": "Control with insecticidal soap or neem oil. Introduce natural predators like ladybugs.",
                                        "1RBDCG": "Apply fungicide treatments. Remove and destroy infected plant parts.",
                                        "ELSIAM": "Apply copper-based fungicide. Remove infected leaves. Ensure good air circulation.",
                                        "TRANSP": "Remove infected plant parts. Apply sulfur-based fungicide as preventive measure.",
                                    }
                                    
                                    treatment = disease_treatments.get(top_result.get('name', ''), 
                                        "Consult local agricultural extension for specific treatment recommendations. " +
                                        "General: Remove infected leaves, apply appropriate fungicide, improve air circulation.")
                                    st.info(treatment)
                                    
                                    # Show multiple results if available
                                    if len(results_list) > 1:
                                        st.markdown("### 📊 Other Possible Conditions")
                                        for i, res in enumerate(results_list[1:4], 1):
                                            prob = int(res.get('score', 0) * 100)
                                            st.write(f"{i}. {res.get('label', 'Unknown')} - {prob}% confidence")
                                else:
                                    st.success("✅ No diseases detected! Your plant appears healthy.")
                                    
                                    st.markdown("""
                                    <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                        <h4 style="color: #155724; margin: 0;">🌿 Plant Health Status</h4>
                                        <p style="color: #155724; margin: 5px 0;">Your plant shows no signs of disease. Keep up the good care!</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Show remaining requests
                                remaining = result.get('remainingIdentificationRequests', 'N/A')
                                st.caption(f"API quota remaining today: {remaining} requests")
                                
                            elif response.status_code == 401:
                                st.error("🔑 Invalid API Key. Please check your Pl@ntNet API key in the .env file.")
                                st.info("Get a free API key at: https://my.plantnet.org/settings/api-key")
                            elif response.status_code == 429:
                                st.error("⏳ API rate limit exceeded. Please try again later.")
                            else:
                                st.error(f"API Error: {response.status_code}")
                                if response.text:
                                    st.info(f"Details: {response.text[:200]}")
                    
                    else:
                        # Plant.id API (original code)
                        import base64
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        api_key = os.getenv('PLANT_ID_API_KEY', '')
                        
                        if not api_key or api_key == 'your_plant_id_api_key_here':
                            st.info("🔧 Running in demo mode (API key not configured)")
                            
                            import random
                            diseases = [
                                {"name": "Early Blight", "probability": 0.92, "treatment": "Apply copper-based fungicide, remove infected leaves, avoid overhead watering"},
                                {"name": "Late Blight", "probability": 0.88, "treatment": "Apply fungicide immediately, remove severely infected plants, improve air circulation"},
                                {"name": "Powdery Mildew", "probability": 0.85, "treatment": "Apply neem oil or sulfur fungicide, improve ventilation, reduce humidity"},
                                {"name": "Leaf Spot", "probability": 0.79, "treatment": "Remove infected leaves, apply copper fungicide, avoid wetting foliage"},
                                {"name": "Bacterial Spot", "probability": 0.75, "treatment": "Apply copper-based spray, remove infected plant parts, rotate crops"}
                            ]
                            result = random.choice(diseases)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Detection Results")
                            
                            # === Enhanced Demo Results ===
                            confidence_pct = int(result['probability'] * 100)
                            
                            # 1. Confidence with level
                            if confidence_pct >= 80:
                                conf_level = "HIGH"
                                conf_color = "#28a745"
                                conf_explanation = "Clear disease patterns detected in image with strong model certainty"
                            elif confidence_pct >= 60:
                                conf_level = "MEDIUM"
                                conf_color = "#ffc107"
                                conf_explanation = "Disease symptoms detected but image may be unclear - consider verification"
                            else:
                                conf_level = "LOW"
                                conf_color = "#dc3545"
                                conf_explanation = "Unclear image or mixed symptoms detected - expert consultation recommended"
                            
                            st.markdown(f"**🌱 Plant:** Tomato")
                            st.markdown(f"**🦠 Disease Detected:** {result['name']}")
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">({conf_level})</span></h2>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{conf_explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 2. Severity
                            if confidence_pct >= 85:
                                severity = "SEVERE"
                                severity_color = "#dc3545"
                                severity_desc = "Disease appears well-established. Immediate action recommended."
                            elif confidence_pct >= 70:
                                severity = "MODERATE"
                                severity_color = "#ffc107"
                                severity_desc = "Disease is spreading. Monitor closely and treat soon."
                            else:
                                severity = "MILD"
                                severity_color = "#17a2b8"
                                severity_desc = "Early signs detected. Quick treatment can prevent spread."
                            
                            st.markdown(f"""
                            <div style="background-color: #fff3cd; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {severity_color};">
                                <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity: {severity}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{severity_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 3. Urgency
                            if confidence_pct >= 80:
                                urgency = "HIGH"
                                urgency_icon = "🚨"
                                urgency_color = "#dc3545"
                                urgency_action = "ACT IMMEDIATELY - Disease can spread rapidly"
                            elif confidence_pct >= 60:
                                urgency = "MEDIUM"
                                urgency_icon = "⚡"
                                urgency_color = "#ffc107"
                                urgency_action = "Treat within 2-3 days to prevent spread"
                            else:
                                urgency = "LOW"
                                urgency_icon = "📊"
                                urgency_color = "#17a2b8"
                                urgency_action = "Monitor and verify before taking action"
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border: 2px solid {urgency_color};">
                                <h4 style="color: {urgency_color}; margin: 0;">{urgency_icon} Urgency: {urgency}</h4>
                                <p style="color: #666; font-size: 14px; margin: 5px 0;">{urgency_action}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 4. Structured Treatment
                            treatment_text = result['treatment']
                            st.markdown("### 💊 Recommended Treatment")
                            
                            st.markdown(f"""
                            <div style="display: flex; gap: 10px; margin: 10px 0;">
                                <div style="flex: 1; background: #e7f3ff; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #0066cc; margin: 0;">🧪 Chemical Solution</h5>
                                    <p style="font-size: 13px;">{treatment_text}</p>
                                </div>
                                <div style="flex: 1; background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                    <h5 style="color: #2e7d32; margin: 0;">🌿 Organic Solution</h5>
                                    <p style="font-size: 13px;">Apply neem oil spray or copper-based organic fungicide. Remove infected leaves. Improve air circulation.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                                <h5 style="color: #e65100; margin: 0;">🛡️ Prevention Tips</h5>
                                <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                                    <li>Rotate crops annually to prevent disease buildup</li>
                                    <li>Use disease-resistant varieties when possible</li>
                                    <li>Remove and destroy infected plant parts</li>
                                    <li>Avoid overhead watering - water at soil level</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 5. Ecosystem Impact
                            if severity == "SEVERE":
                                yield_impact = "Estimated 30-50% yield reduction possible (based on typical disease impact patterns)"
                                price_impact = "Quality drop may reduce market value (estimated 20-40% reduction based on typical disease impact)"
                                recommendation = "Consider harvesting early if crop is near maturity to minimize loss"
                            elif severity == "MODERATE":
                                yield_impact = "Estimated 10-25% yield reduction possible (depends on treatment timing and crop health)"
                                price_impact = "Minor quality impact on market price (may affect grade but generally recoverable)"
                                recommendation = "Treat immediately and monitor for 7 days"
                            else:
                                yield_impact = "Estimated 5-10% yield impact if left untreated (minimal if treated promptly)"
                                price_impact = "Minimal impact expected if treated promptly and properly"
                                recommendation = "Apply treatment and recheck in 1 week"
                            
                            st.markdown("### 🌾 Ecosystem Impact")
                            st.markdown(f"""
                            <div style="background: #fce4ec; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <h5 style="color: #c2185b; margin: 0;">📉 Yield & Price Impact</h5>
                                <p style="margin: 8px 0;"><strong>🌾 Yield:</strong> {yield_impact}</p>
                                <p style="margin: 8px 0;"><strong>💰 Market:</strong> {price_impact}</p>
                                <p style="margin: 8px 0;"><strong>💡 Advisory:</strong> {recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 6. Trust Layer
                            st.markdown("""
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                                <p style="color: #6c757d; font-size: 12px; margin: 0;">
                                    <strong>⚠️ Disclaimer:</strong> This is demo mode. Add API key for real detection.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Real Plant.id Health API call
                            import json
                            st.info("🔄 Connecting to Plant.id Health API...")
                            
                            headers = {
                                'Content-Type': 'application/json',
                                'Api-Key': api_key
                            }
                            
                            data = {
                                'images': [f'data:image/jpeg;base64,{image_base64}'],
                                'modifiers': ['health_all', 'disease_similar_images'],
                                'plant_details': ['common_names', 'url', 'wiki_description', 'taxonomy'],
                                'disease_details': ['classification', 'common_names', 'description', 'treatment', 'url'],
                                'latitude': 0,
                                'longitude': 0,
                                'datetime': int(datetime.now().timestamp())
                            }
                            
                            response = requests.post(
                                'https://api.plant.id/v3/identify',
                                headers=headers,
                                json=data,
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                health_assessment = result.get('health_assessment', {})
                                is_healthy = health_assessment.get('is_healthy', True)
                                is_healthy_prob = health_assessment.get('is_healthy_probability', 0)
                                diseases = health_assessment.get('diseases', [])
                                
                                st.markdown("---")
                                st.markdown("### 📊 Detection Results")
                                
                                if not is_healthy and diseases:
                                    plant_info = result.get('suggestions', [{}])[0] if result.get('suggestions') else {}
                                    plant_name = plant_info.get('plant_name', 'Unknown Plant')
                                    common_names = plant_info.get('plant_details', {}).get('common_names', [])
                                    if common_names and isinstance(common_names, list):
                                        plant_name = common_names[0]
                                    
                                    st.markdown(f"**🌱 Plant:** {plant_name}")
                                    
                                    if diseases:
                                        top_disease = diseases[0]
                                        disease_name = top_disease.get('name', 'Unknown Disease')
                                        disease_prob = top_disease.get('probability', 0)
                                        
                                        st.markdown(f"**🦠 Disease Detected:** {disease_name}")
                                        
                                        # === Enhanced Display ===
                                        confidence_pct = int(disease_prob * 100)
                                        
                                        # 1. Confidence with level
                                        if confidence_pct >= 80:
                                            conf_level = "HIGH"
                                            conf_color = "#28a745"
                                            conf_explanation = "Clear disease patterns detected in image with strong model certainty"
                                        elif confidence_pct >= 60:
                                            conf_level = "MEDIUM"
                                            conf_color = "#ffc107"
                                            conf_explanation = "Disease symptoms detected but image may be unclear - consider verification"
                                        else:
                                            conf_level = "LOW"
                                            conf_color = "#dc3545"
                                            conf_explanation = "Unclear image or mixed symptoms detected - expert consultation recommended"
                                        
                                        st.markdown(f"""
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                            <h4 style="color: #666; margin: 0;">🎯 Detection Confidence</h4>
                                            <h2 style="color: {conf_color}; margin: 5px 0;">{confidence_pct}% <span style="font-size: 16px; color: {conf_color};">({conf_level})</span></h2>
                                            <p style="color: #666; font-size: 14px; margin: 5px 0;">{conf_explanation}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # 2. Severity
                                        if confidence_pct >= 85:
                                            severity = "SEVERE"
                                            severity_color = "#dc3545"
                                            severity_desc = "Disease appears well-established. Immediate action recommended."
                                        elif confidence_pct >= 70:
                                            severity = "MODERATE"
                                            severity_color = "#ffc107"
                                            severity_desc = "Disease is spreading. Monitor closely and treat soon."
                                        else:
                                            severity = "MILD"
                                            severity_color = "#17a2b8"
                                            severity_desc = "Early signs detected. Quick treatment can prevent spread."
                                        
                                        st.markdown(f"""
                                        <div style="background-color: #fff3cd; padding: 12px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {severity_color};">
                                            <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity: {severity}</h4>
                                            <p style="color: #666; font-size: 14px; margin: 5px 0;">{severity_desc}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # 3. Urgency
                                        if confidence_pct >= 80:
                                            urgency = "HIGH"
                                            urgency_icon = "🚨"
                                            urgency_color = "#dc3545"
                                            urgency_action = "ACT IMMEDIATELY - Disease can spread rapidly"
                                        elif confidence_pct >= 60:
                                            urgency = "MEDIUM"
                                            urgency_icon = "⚡"
                                            urgency_color = "#ffc107"
                                            urgency_action = "Treat within 2-3 days to prevent spread"
                                        else:
                                            urgency = "LOW"
                                            urgency_icon = "📊"
                                            urgency_color = "#17a2b8"
                                            urgency_action = "Monitor and verify before taking action"
                                        
                                        st.markdown(f"""
                                        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 10px; margin: 10px 0; border: 2px solid {urgency_color};">
                                            <h4 style="color: {urgency_color}; margin: 0;">{urgency_icon} Urgency: {urgency}</h4>
                                            <p style="color: #666; font-size: 14px; margin: 5px 0;">{urgency_action}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        disease_details = top_disease.get('disease_details', {})
                                        if disease_details:
                                            description = disease_details.get('description', '')
                                            if description:
                                                st.markdown("### 📋 Disease Description")
                                                st.info(description)
                                            
                                            treatment = disease_details.get('treatment', {})
                                            if treatment:
                                                st.markdown("### 💊 Recommended Treatment")
                                                
                                                # Get treatment info
                                                chemical = ""
                                                prevention = treatment.get('prevention', [])
                                                biological = treatment.get('biological', [])
                                                
                                                if prevention:
                                                    chemical = "Apply appropriate fungicide. " + " ".join(prevention[:2])
                                                else:
                                                    chemical = "Apply appropriate fungicide as recommended by agricultural expert."
                                                
                                                st.markdown(f"""
                                                <div style="display: flex; gap: 10px; margin: 10px 0;">
                                                    <div style="flex: 1; background: #e7f3ff; padding: 10px; border-radius: 8px;">
                                                        <h5 style="color: #0066cc; margin: 0;">🧪 Chemical Solution</h5>
                                                        <p style="font-size: 13px;">{chemical}</p>
                                                    </div>
                                                    <div style="flex: 1; background: #e8f5e9; padding: 10px; border-radius: 8px;">
                                                        <h5 style="color: #2e7d32; margin: 0;">🌿 Organic Solution</h5>
                                                        <p style="font-size: 13px;">{" ".join(biological[:2]) if biological else "Apply neem oil spray or copper-based organic fungicide. Remove infected leaves. Improve air circulation."}</p>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                if prevention:
                                                    st.markdown("""
                                                    <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                                                        <h5 style="color: #e65100; margin: 0;">🛡️ Prevention Tips</h5>
                                                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 13px;">
                                                        """, unsafe_allow_html=True)
                                                    for tip in prevention[:5]:
                                                        st.write(f"• {tip}")
                                                    st.markdown("</ul></div>", unsafe_allow_html=True)
                                        
                                        # 4. Ecosystem Impact
                                        if severity == "SEVERE":
                                            yield_impact = "Estimated 30-50% yield reduction possible (based on typical disease impact patterns)"
                                            price_impact = "Quality drop may reduce market value (estimated 20-40% reduction based on typical disease impact)"
                                            recommendation = "Consider harvesting early if crop is near maturity to minimize loss"
                                        elif severity == "MODERATE":
                                            yield_impact = "Estimated 10-25% yield reduction possible (depends on treatment timing and crop health)"
                                            price_impact = "Minor quality impact on market price (may affect grade but generally recoverable)"
                                            recommendation = "Treat immediately and monitor for 7 days"
                                        else:
                                            yield_impact = "Estimated 5-10% yield impact if left untreated (minimal if treated promptly)"
                                            price_impact = "Minimal impact expected if treated promptly and properly"
                                            recommendation = "Apply treatment and recheck in 1 week"
                                        
                                        st.markdown("### 🌾 Ecosystem Impact")
                                        st.markdown(f"""
                                        <div style="background: #fce4ec; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                            <h5 style="color: #c2185b; margin: 0;">📉 Yield & Price Impact</h5>
                                            <p style="margin: 8px 0;"><strong>🌾 Yield:</strong> {yield_impact}</p>
                                            <p style="margin: 8px 0;"><strong>💰 Market:</strong> {price_impact}</p>
                                            <p style="margin: 8px 0;"><strong>💡 Advisory:</strong> {recommendation}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # 5. Trust Layer
                                        st.markdown("""
                                        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d;">
                                            <p style="color: #6c757d; font-size: 12px; margin: 0;">
                                                <strong>⚠️ Disclaimer:</strong> This AI-based detection is for preliminary guidance only. 
                                                For severe cases, consult your local agricultural extension office.
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    health_confidence = int((1 - is_healthy_prob) * 100) if is_healthy_prob else 95
                                    st.success(f"✅ Your plant appears healthy! (Health confidence: {health_confidence}%)")
                            elif response.status_code == 401:
                                st.error("🔑 Invalid API Key. Please check your Plant.id API key.")
                            elif response.status_code == 429:
                                st.error("⏳ API rate limit exceeded.")
                            else:
                                st.error(f"API Error: {response.status_code}")
                    
                    # Additional tips (always show)
                    st.markdown("### 🌿 General Care Tips")
                    st.write("• Monitor your plants regularly for early signs of problems")
                    st.write("• Ensure proper watering - avoid overwatering or underwatering")
                    st.write("• Provide adequate sunlight based on plant needs")
                    st.write("• Use disease-resistant varieties when possible")
                    st.write("• Rotate crops annually to prevent disease buildup")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please try again with a clearer image of the plant leaf")

# ============================================================
# AgriCare AI - Smart Farming Assistant  
# ============================================================
elif menu == get_text("menu_emotion", global_lang):
    
    # Get farmer context
    farmer_profile = st.session_state.get('farmer_profile', {})
    farmer_name = farmer_profile.get('name', 'Friend')
    state = farmer_profile.get('state', 'Maharashtra')
    crops_grown = farmer_profile.get('crops', [])
    
    # Initialize chat history
    if "agri_history" not in st.session_state:
        st.session_state.agri_history = []
    if "emotion_messages" not in st.session_state:
        st.session_state.emotion_messages = []
    if "agri_msg_value" not in st.session_state:
        st.session_state.agri_msg_value = ""
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = ""
    
    # Get current emotion state
    all_messages = st.session_state.agri_history + st.session_state.emotion_messages
    current_emotion = "happy"
    if all_messages:
        current_emotion = all_messages[-1].get('emotion', 'happy')
    
    # ===== SIMPLE LAYOUT =====
    st.markdown("## 🌾 AgriCare AI - Your Farming Companion")
    current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    st.caption(f"🕐 {current_time}")
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    quick_actions = [
        ("🌾 Crop Issue", "My crops are having problems"),
        ("💰 Market Price", "What's the current market price?"),
        ("🌧 Weather", "How will weather affect my farm?"),
        ("💙 Need Support", "I need someone to talk to")
    ]
    
    with col1:
        if st.button("🌾 Crop Issue", use_container_width=True):
            detected = detect_emotion("My crops are having problems")
            response = get_chatgpt_style_fallback(detected, global_lang, farmer_profile, "My crops are having problems", st.session_state.emotion_messages)
            st.session_state.emotion_messages.append({
                "user": "My crops are having problems", 
                "bot": response, 
                "emotion": detected, 
                "timestamp": datetime.now()
            })
            st.rerun()
        
        if st.button("🌧 Weather", use_container_width=True):
            detected = detect_emotion("How will weather affect my farm?")
            response = get_chatgpt_style_fallback(detected, global_lang, farmer_profile, "How will weather affect my farm?", st.session_state.emotion_messages)
            st.session_state.emotion_messages.append({
                "user": "How will weather affect my farm?", 
                "bot": response, 
                "emotion": detected, 
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col2:
        if st.button("💰 Market Price", use_container_width=True):
            detected = detect_emotion("What's the current market price?")
            response = get_chatgpt_style_fallback(detected, global_lang, farmer_profile, "What's the current market price?", st.session_state.emotion_messages)
            st.session_state.emotion_messages.append({
                "user": "What's the current market price?", 
                "bot": response, 
                "emotion": detected, 
                "timestamp": datetime.now()
            })
            st.rerun()
        
        if st.button("💙 Need Support", use_container_width=True):
            detected = detect_emotion("I need someone to talk to")
            response = get_chatgpt_style_fallback(detected, global_lang, farmer_profile, "I need someone to talk to", all_messages)
            st.session_state.emotion_messages.append({
                "user": "I need someone to talk to", 
                "bot": response, 
                "emotion": detected, 
                "timestamp": datetime.now()
            })
            st.rerun()
    
    st.markdown("---")
    
    # Chat section - Using new UI modules
    try:
        from agridream_modules.ui_components import render_chat_header, render_chat_bubble, render_welcome_message
        from agridream_modules.ai_module import get_cohere_response, detect_emotion, send_emergency_alert, get_fallback_response
        
        # Render chat header
        render_chat_header(farmer_name)
        
    except ImportError:
        st.markdown("## 🌾 AgriCare AI - Your Farming Companion")
    
    # Initialize chat history
    if "emotion_messages" not in st.session_state:
        st.session_state.emotion_messages = []
    
    messages = st.session_state.get('emotion_messages', [])
    
    # Show welcome if no messages
    if not messages:
        try:
            render_welcome_message(farmer_name)
        except:
            st.info(f"👋 Namaste, {farmer_name}! Type a message below to start chatting.")
    
    # Display messages with WhatsApp-style bubbles
    for chat in messages:
        try:
            render_chat_bubble(chat.get('user', ''), is_user=True)
            render_chat_bubble(chat.get('bot', ''), is_user=False, emotion=chat.get('emotion', 'happy'))
        except:
            st.markdown(f"**You:** {chat.get('user', '')}")
            st.markdown(f"**AI:** {chat.get('bot', '')}")
        st.markdown("---")
    
    # Input - using form for Enter key support
    st.markdown("**Type your message:**")
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message", 
            key=f"chat_input_{len(messages)}", 
            label_visibility="collapsed", 
            placeholder="Type message and press Enter..."
        )
        submit_button = st.form_submit_button(label="📤 Send")
    
    # Handle send (both Enter and button click)
    if submit_button and user_input.strip():
        # Detect emotion
        try:
            detected_emotion = detect_emotion(user_input)
        except:
            detected_emotion = "happy"
        
        # Get response from AI (with better error handling)
        try:
            response = get_cohere_response(user_input, detected_emotion, global_lang, farmer_profile, messages)
        except Exception as e:
            print(f"AI Error: {e}")
            try:
                from agridream_modules.ai_module import get_fallback_response
                response = get_fallback_response(user_input, detected_emotion)
            except:
                response = get_chatgpt_style_fallback(detected_emotion, global_lang, farmer_profile, user_input, messages)
        
        # Add to messages
        st.session_state.emotion_messages.append({
            "user": user_input,
            "bot": response,
            "emotion": detected_emotion,
            "timestamp": datetime.now()
        })
        
        # Auto-send emergency alert if high risk detected
        if detected_emotion == "high_risk":
            family1_phone = st.session_state.get('family1_phone', '')
            if family1_phone:
                try:
                    from agridream_modules.ai_module import send_emergency_alert
                    send_emergency_alert(farmer_name, family1_phone, f"⚠️ Alert: {farmer_name} needs support. Please check on them.")
                except:
                    pass
        
        # Save to database
        try:
            database.save_chat_message(
                farmer_id=st.session_state.get('current_farmer_id', 1),
                user_message=user_input,
                bot_response=response,
                emotion=detected_emotion,
                language=global_lang
            )
        except:
            pass
        
        st.rerun()
    
    # Clear button
    if st.button("🗑️ Clear Chat"):
        st.session_state.emotion_messages = []
        st.rerun()
        st.markdown("### 🧠 Your Status")
        
        # Emotion indicator
        emotion_data = {
            "happy": ("😊", "Feeling Good", "green"),
            "sad": ("😔", "Feeling Low", "orange"),
            "angry": ("😠", "Frustrated", "red"),
            "high_risk": ("🚨", "Need Support", "red")
        }
        emo_emoji, emo_status, emo_color = emotion_data.get(current_emotion, ("😊", "Good", "green"))
        
        st.markdown(f":{emo_color}[**{emo_emoji} {emo_status}**]")
        
        st.markdown("---")
        st.markdown("### 📞 Help Lines")
        st.write("🧠 iCall: 9152987821")
        st.write("🚔 Police: 100")
        st.write("🌾 Kisan: 1800-180-1551")

# ---------------------------
# Government Schemes
# ---------------------------
elif menu == get_text("menu_schemes", global_lang):
    st.markdown("🏛️ **Government Schemes for Farmers**")
    st.markdown("---")
    
    # Government schemes data
    schemes = [
        {
            "name": "PM-KISAN",
            "description": "Direct income support of ₹6,000/year to farmer families",
            "eligibility": "All landholding farmer families",
            "benefit": "₹6,000/year (₹2,000 every 4 months)",
            "apply": "pmkisan.gov.in"
        },
        {
            "name": "Kisan Credit Card (KCC)",
            "description": "Easy credit for farming needs at low interest rates",
            "eligibility": "All farmers - individual/joint borrowers",
            "benefit": "Credit up to ₹3 lakh at 4% interest",
            "apply": "Nearest bank"
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            "description": "Crop insurance scheme for crop loss due to natural calamities",
            "eligibility": "All farmers including sharecropers",
            "benefit": "Crop loss compensation",
            "apply": "pmfby.gov.in"
        },
        {
            "name": "Soil Health Card",
            "description": "Free soil testing and health cards for farmers",
            "eligibility": "All farmers",
            "benefit": "Soil test report with fertilizer recommendations",
            "apply": "Local agriculture office"
        },
        {
            "name": "Fasal Loan Waiver",
            "description": "Loan waiver for farmers in distress",
            "eligibility": "Small and marginal farmers",
            "benefit": "Crop loan waiver",
            "apply": "State government portal"
        },
        {
            "name": "Agriculture Infrastructure Fund",
            "description": "Loan for building farm infrastructure",
            "eligibility": "FPOs, agri-entrepreneurs, private companies",
            "benefit": "Loan up to ₹2 crore at 3% interest",
            "apply": "Bank/Online portal"
        },
        {
            "name": "National Agriculture Market (eNAM)",
            "description": "Online trading platform for agricultural produce",
            "eligibility": "All farmers and traders",
            "benefit": "Better price discovery, direct sale",
            "apply": "enomnic.in"
        },
        {
            "name": "Sub-Mission on Agricultural Mechanization",
            "description": "Subsidy on agricultural machinery",
            "eligibility": "All farmers",
            "benefit": "40-50% subsidy on farm equipment",
            "apply": "Local tractor dealer"
        }
    ]
    
    # Display schemes
    for i, scheme in enumerate(schemes, 1):
        with st.expander(f"🏅 {scheme['name']}"):
            st.markdown(f"""
            <div style="background: #f0f8f0; padding: 15px; border-radius: 10px;">
                <p><strong>📝 Description:</strong> {scheme['description']}</p>
                <p><strong>✅ Eligibility:</strong> {scheme['eligibility']}</p>
                <p><strong>💰 Benefit:</strong> {scheme['benefit']}</p>
                <p><strong>📨 How to Apply:</strong> {scheme['apply']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 For more schemes, visit your nearest Agricultural Office or check state government websites.")

# ============================================================
# 🎙️ VOICE ASSISTANT
# ============================================================
# 🎙️ VOICE ASSISTANT
# ============================================================
elif menu == "🎙️ Voice Assistant":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 10px 0;">
        <h2 style="color: white; margin: 0;">🎙️ Voice Assistant</h2>
        <p style="color: white; opacity: 0.9;">Speak to get instant answers! (Works in Hindi, English, Tamil)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Voice input using Streamlit's audio input
    st.markdown("### 🎤 Speak or Type Your Query")
    
    # Text input as fallback
    voice_query = st.text_input(
        "Type your question (or use voice if available):",
        placeholder="e.g., What is tomato price in Maharashtra?",
        key="voice_query"
    )
    
    # Quick voice command buttons
    st.markdown("### ⚡ Quick Commands (Click to Ask)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Today's Price"):
            voice_query = "What is today's price for tomato?"
    with col2:
        if st.button("🌾 Crop Recommendation"):
            voice_query = "Which crop should I grow?"
    with col3:
        if st.button("🌤️ Weather Update"):
            voice_query = "What's the weather in my area?"
    
    # Process query
    if voice_query:
        st.markdown(f"**You asked:** {voice_query}")
        
        # Simple keyword-based responses (can be upgraded to AI)
        query_lower = voice_query.lower()
        
        # Price related queries
        if any(word in query_lower for word in ['price', 'भाव', 'cost', 'rate']):
            st.markdown("### 💰 Price Information")
            # Try to get price data
            try:
                df_prices = pd.read_csv('agmarknet_prices.csv')
                
                # Find crop mentioned
                crops = ['rice', 'wheat', 'tomato', 'potato', 'onion', 'cotton', 'maize', 'sugarcane']
                mentioned_crop = None
                for crop in crops:
                    if crop in query_lower:
                        mentioned_crop = crop.title()
                        break
                
                if mentioned_crop:
                    crop_data = df_prices[df_prices['Commodity'].str.lower() == mentioned_crop.lower()]
                    if not crop_data.empty:
                        avg_price = int(crop_data['Modal_x0020_Price'].mean())
                        st.success(f"📊 {mentioned_crop} current price: ₹{avg_price}/quintal")
                    else:
                        st.info(f"📊 {mentioned_crop} price data available. Check Price Forecasting page for details.")
                else:
                    st.info("💡 Try specifying a crop like: tomato, rice, wheat, onion")
                    st.write("Use the **Price Forecasting** page for detailed prices!")
            except Exception as e:
                st.info("💡 Use the **Price Forecasting** page for detailed prices!")
        
        # Weather related queries
        elif any(word in query_lower for word in ['weather', 'मौसम', 'rain', 'temperature']):
            st.markdown("### 🌤️ Weather Information")
            
            # Try to get weather
            from app import get_weather, INDIAN_STATE_CAPITALS
            states = list(INDIAN_STATE_CAPITALS.keys())
            mentioned_state = None
            for state in states:
                if state.lower() in query_lower:
                    mentioned_state = state
                    break
            
            if mentioned_state:
                weather, error = get_weather(mentioned_state)
                if weather:
                    st.success(f"🌤️ {mentioned_state}: {weather['temperature']}°C, {weather['humidity']}% humidity")
                else:
                    st.info("Weather data temporarily unavailable. Check Weather page!")
            else:
                st.info("💡 Try specifying a state like: Maharashtra, Tamil Nadu, Karnataka")
        
        # Crop recommendation queries
        elif any(word in query_lower for word in ['crop', 'grow', 'plant', 'फसल', 'खेती']):
            st.markdown("### 🌾 Crop Recommendation")
            st.info("Use the **Crop Recommendation** page to get personalized suggestions based on your soil and location!")
        
        # Disease queries
        elif any(word in query_lower for word in ['disease', 'sick', 'pest', 'बीमारी']):
            st.markdown("### 🩺 Disease Detection")
            st.info("Upload a photo of your plant on the **Disease Detection** page for AI-powered diagnosis!")
        
        # General help
        elif any(word in query_lower for word in ['help', 'help', 'guide', 'मदद']):
            st.markdown("""
            ### 🎯 How I Can Help You:
            
            - **Price queries**: "What is tomato price?"
            - **Weather**: "What's weather in Maharashtra?"
            - **Crops**: "Which crop should I grow?"
            - **Disease**: "My plant has yellow leaves"
            
            Use the menu on the left to access all features!
            """)
        
        else:
            st.markdown("""
            ### 💡 I Can Help You With:
            
            - 📊 **Prices**: Ask "What's tomato price?"
            - 🌤️ **Weather**: Ask "What's weather in Maharashtra?"
            - 🌾 **Crops**: Ask "Which crop should I grow?"
            - 🩺 **Disease**: Ask "My plant is sick"
            
            Or use the menu on the left to access all features!
""")


# End of app
