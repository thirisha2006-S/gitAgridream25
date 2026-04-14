"""
Populate database with realistic sample data
Run this once to add price data for testing
"""

import database
from datetime import datetime, timedelta
import random

def populate_sample_data():
    """Populate database with realistic sample data"""
    
    print("Populating sample data...")
    
    # Get crop IDs
    crops_data = database.get_all_crops()
    crop_map = {c['name']: c['id'] for c in crops_data}
    print(f"Available crops: {list(crop_map.keys())}")
    
    # Sample markets for different states
    markets = [
        ("Pune APMC", "Maharashtra", "Pune"),
        ("Mumbai APMC", "Maharashtra", "Mumbai"),
        ("Nagpur APMC", "Maharashtra", "Nagpur"),
        ("Nashik APMC", "Maharashtra", "Nashik"),
        ("Chennai Koyambedu", "Tamil Nadu", "Chennai"),
        ("Coimbatore Market", "Tamil Nadu", "Coimbatore"),
        ("Madurai Market", "Tamil Nadu", "Madurai"),
        ("Delhi Azadpur", "Delhi", "North Delhi"),
        ("Lucknow Market", "Uttar Pradesh", "Lucknow"),
        ("Kolkata Market", "West Bengal", "Kolkata"),
        ("Bengaluru Market", "Karnataka", "Bengaluru"),
        ("Ahmedabad Market", "Gujarat", "Ahmedabad"),
    ]
    
    # Insert markets
    market_ids = {}
    for name, state, district in markets:
        mid = database.insert_market(name, state, district)
        market_ids[(name, state)] = mid
    print(f"Added {len(markets)} markets")
    
    # Generate price data for the last 60 days
    base_date = datetime.now()
    
    price_configs = {
        "Rice": {"base": 2200, "range": 200},
        "Wheat": {"base": 2400, "range": 150},
        "Cotton": {"base": 6200, "range": 400},
        "Tomato": {"base": 1800, "range": 800},
        "Potato": {"base": 1500, "range": 200},
        "Onion": {"base": 1600, "range": 400},
        "Maize": {"base": 1700, "range": 150},
        "Sugarcane": {"base": 3800, "range": 200},
    }
    
    total_prices = 0
    
    # For each crop and state combination
    for crop_name, config in price_configs.items():
        if crop_name not in crop_map:
            continue
            
        crop_id = crop_map[crop_name]
        
        # Generate prices for Maharashtra, Tamil Nadu, Delhi
        states_markets = {
            "Maharashtra": ["Pune APMC", "Mumbai APMC", "Nagpur APMC"],
            "Tamil Nadu": ["Chennai Koyambedu", "Coimbatore Market"],
            "Delhi": ["Delhi Azadpur"],
            "Uttar Pradesh": ["Lucknow Market"],
            "West Bengal": ["Kolkata Market"],
            "Karnataka": ["Bengaluru Market"],
            "Gujarat": ["Ahmedabad Market"],
        }
        
        for state, market_names in states_markets.items():
            for market_name in market_names:
                market_key = (market_name, state)
                if market_key not in market_ids:
                    continue
                
                market_id = market_ids[market_key]
                
                # Generate 60 days of price data
                for day_offset in range(60):
                    date = base_date - timedelta(days=day_offset)
                    date_str = date.strftime("%Y-%m-%d")
                    
                    # Add some variation to price
                    variation = random.uniform(-config["range"], config["range"])
                    seasonal = random.uniform(-50, 50)  # Seasonal fluctuation
                    modal_price = config["base"] + variation + seasonal
                    
                    min_price = modal_price * 0.9
                    max_price = modal_price * 1.1
                    arrivals = random.randint(50, 500)
                    
                    try:
                        database.insert_price(
                            crop_id=crop_id,
                            market_id=market_id,
                            date=date_str,
                            modal_price=round(modal_price, 2),
                            min_price=round(min_price, 2),
                            max_price=round(max_price, 2),
                            arrivals=arrivals
                        )
                        total_prices += 1
                    except Exception as e:
                        pass  # Skip duplicates
    
    print(f"Added {total_prices} price records")
    
    # Add some farmer profiles for testing
    farmers = [
        ("Ramesh Patil", "+919999999999", "Maharashtra", "Pune", "Rice", "Large"),
        ("Karim Khan", "+919888888888", "Maharashtra", "Nagpur", "Cotton", "Medium"),
        ("John Peter", "+919877777777", "Tamil Nadu", "Chennai", "Rice", "Small"),
    ]
    
    for name, phone, state, district, main_crop, farm_size in farmers:
        try:
            database.insert_farmer(
                name=name, phone=phone, state=state, district=district,
                main_crop=main_crop, farm_size=farm_size
            )
        except:
            pass
    
    print("Sample data populated successfully!")
    
    # Show final stats
    stats = database.get_database_stats()
    print(f"\nFinal Database Stats:")
    for table, count in stats.items():
        print(f"  {table}: {count}")

if __name__ == "__main__":
    # Initialize database first
    database.initialize_database()
    
    # Then populate sample data
    populate_sample_data()
