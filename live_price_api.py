#!/usr/bin/env python3
"""
Live agricultural price API using Yahoo Finance and other sources
"""

import requests
import json
from datetime import datetime

def get_yahoo_commodity_price(symbol, name):
    """Get commodity price from Yahoo Finance"""
    try:
        # Yahoo Finance unofficial API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})

                price = meta.get('regularMarketPrice')
                previous_close = meta.get('previousClose', 0)

                if price:
                    change = price - previous_close if previous_close else 0
                    change_percent = (change / previous_close * 100) if previous_close else 0

                    return {
                        'commodity': name,
                        'price': price,
                        'change': change,
                        'change_percent': change_percent,
                        'currency': 'USD',
                        'source': 'Yahoo Finance',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }

        return None

    except Exception as e:
        print(f"Yahoo Finance error for {symbol}: {e}")
        return None

def get_alpha_vantage_price(symbol, name, api_key=None):
    """Get commodity price from Alpha Vantage"""
    try:
        if not api_key:
            api_key = "demo"  # Free tier

        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if 'Global Quote' in data:
                quote = data['Global Quote']
                price = quote.get('05. price')
                change = quote.get('09. change')
                change_percent = quote.get('10. change percent')

                if price and price != 'None':
                    return {
                        'commodity': name,
                        'price': float(price),
                        'change': float(change) if change else 0,
                        'change_percent': float(change_percent.strip('%')) if change_percent else 0,
                        'currency': 'USD',
                        'source': 'Alpha Vantage',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }

        return None

    except Exception as e:
        print(f"Alpha Vantage error for {symbol}: {e}")
        return None

def get_commodity_price_live(commodity):
    """Get live commodity price from various sources"""
    try:
        commodity = commodity.lower().strip()

        # Map commodities to Yahoo Finance symbols
        commodity_map = {
            'wheat': ('WHEAT', 'Wheat'),
            'corn': ('CORN', 'Corn'),
            'soybean': ('SOY', 'Soybean'),
            'rice': ('RR', 'Rice'),
            'cotton': ('CT', 'Cotton'),
            'sugar': ('SB', 'Sugar'),
            'coffee': ('KC', 'Coffee'),
            'cocoa': ('CC', 'Cocoa'),
            'gold': ('GC=F', 'Gold'),
            'silver': ('SI=F', 'Silver'),
            'crude oil': ('CL=F', 'Crude Oil'),
            'copper': ('HG=F', 'Copper')
        }

        # Agricultural commodities mapping
        agri_commodities = {
            'wheat': ('KE=F', 'KC Wheat'),  # Kansas City Wheat
            'corn': ('ZC=F', 'Corn'),
            'soybean': ('ZS=F', 'Soybean'),
            'rice': ('ZR=F', 'Rice'),
            'cotton': ('CT=F', 'Cotton'),
            'sugar': ('SB=F', 'Sugar'),
            'coffee': ('KC=F', 'Coffee'),
            'cocoa': ('CC=F', 'Cocoa'),
            'orange juice': ('OJ=F', 'Orange Juice'),
            'lumber': ('LBS=F', 'Lumber')
        }

        # Check if we have a mapping for this commodity
        if commodity in agri_commodities:
            symbol, name = agri_commodities[commodity]

            # Try Yahoo Finance first
            result = get_yahoo_commodity_price(symbol, name)
            if result:
                return result

            # Try Alpha Vantage as backup
            result = get_alpha_vantage_price(symbol, name)
            if result:
                return result

        # For Indian agricultural commodities, try international equivalents
        indian_to_global = {
            'rice': 'ZR=F',  # Rough Rice
            'wheat': 'KE=F', # KC Wheat
            'maize': 'ZC=F', # Corn
            'cotton': 'CT=F',
            'sugarcane': 'SB=F', # Sugar
            'soybean': 'ZS=F',
            'groundnut': 'None',  # No direct equivalent
            'tomato': 'None',     # No direct equivalent
            'potato': 'None',     # No direct equivalent
            'onion': 'None'       # No direct equivalent
        }

        if commodity in indian_to_global and indian_to_global[commodity] != 'None':
            symbol = indian_to_global[commodity]
            name = f"{commodity.title()} (Global)"

            result = get_yahoo_commodity_price(symbol, name)
            if result:
                # Convert USD to INR (approximate)
                usd_to_inr = 83.0  # Current approximate rate
                result['price_inr'] = result['price'] * usd_to_inr
                result['currency'] = 'INR'
                result['note'] = 'Global commodity price converted to INR'
                return result

        return None

    except Exception as e:
        print(f"Live price API error: {e}")
        return None

def format_price_response(result):
    """Format the price result for display"""
    if not result:
        return "No live price data available"

    try:
        commodity = result['commodity']
        price = result['price']
        change = result['change']
        change_percent = result['change_percent']
        currency = result['currency']
        source = result['source']

        # Format the response
        response = f"📈 LIVE {commodity} Price\n"
        response += f"💰 Current Price: {currency} {price:.2f}\n"

        if change != 0:
            change_symbol = "📈" if change > 0 else "📉"
            response += f"{change_symbol} Change: {change:+.2f} ({change_percent:+.2f}%)\n"

        response += f"🔍 Source: {source}\n"

        if 'price_inr' in result:
            response += f"🇮🇳 INR Equivalent: ₹{result['price_inr']:.0f}\n"

        if 'note' in result:
            response += f"ℹ️ {result['note']}\n"

        response += f"🕒 Last Updated: {result['timestamp'][:16].replace('T', ' ')}"

        return response

    except Exception as e:
        return f"Error formatting price data: {e}"

def test_live_prices():
    """Test the live price functionality"""
    print("Testing Live Agricultural Price API")
    print("=" * 40)

    test_commodities = ['wheat', 'corn', 'soybean', 'rice', 'cotton', 'sugar']

    for commodity in test_commodities:
        print(f"\n--- Testing {commodity} ---")
        result = get_commodity_price_live(commodity)
        if result:
            formatted = format_price_response(result)
            print(formatted)
        else:
            print(f"No live price data found for {commodity}")

if __name__ == "__main__":
    test_live_prices()