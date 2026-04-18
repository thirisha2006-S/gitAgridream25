"""
AgriDream Database Module v2.0
==============================
Enhanced database layer with decision system support.
Optimized for read-heavy operations with proper error handling.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import random

# Database file path - use absolute path to ensure consistency
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "agridream.db")


# ============================================================
# 1. CONNECTION MANAGEMENT
# ============================================================

def get_connection() -> sqlite3.Connection:
    """Get SQLite database connection."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise


# ============================================================
# 2. TABLE CREATION (ENHANCED)
# ============================================================

def create_tables():
    """Create all required tables with enhanced fields."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Chat History Table for AgriCare AI
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                emotion TEXT,
                language TEXT DEFAULT 'English',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            )
        """)
        
        # Farmers Table (ENHANCED - Context Aware)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT,
                state TEXT NOT NULL,
                district TEXT,
                language TEXT DEFAULT 'English',
                main_crop TEXT,
                farm_size TEXT DEFAULT 'Medium',
                preferred_market TEXT,
                emergency_contact TEXT,
                risk_preference TEXT DEFAULT 'Medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Crops Table (ENHANCED)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                season TEXT,
                category TEXT,
                avg_growth_days INTEGER,
                min_price_expected INTEGER,
                max_price_expected INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Markets Table (ENHANCED - for Best vs Nearby logic)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, state, district)
            )
        """)
        
        # Prices Table (CORE - with proper date tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER NOT NULL,
                market_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                modal_price REAL NOT NULL,
                min_price REAL,
                max_price REAL,
                arrivals INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crop_id) REFERENCES crops(id),
                FOREIGN KEY (market_id) REFERENCES markets(id),
                UNIQUE(crop_id, market_id, date)
            )
        """)
        
        # Create indexes for fast reads (IMPORTANT)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_crop ON prices(crop_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_crop_date ON prices(crop_id, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_crop_state ON prices(crop_id)")
        
        # Predictions Table (ENHANCED - with decision reasoning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER,
                crop TEXT NOT NULL,
                state TEXT NOT NULL,
                current_price REAL,
                predicted_price REAL,
                trend TEXT NOT NULL,
                confidence TEXT NOT NULL,
                decision TEXT NOT NULL,
                decision_reason TEXT NOT NULL,
                money_impact TEXT,
                model_used TEXT DEFAULT 'Moving Average',
                data_quality REAL DEFAULT 100.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_farmer ON predictions(farmer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at)")
        
        # Alerts Table (NEW - for proactive notifications)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER,
                alert_type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT DEFAULT 'Medium',
                is_read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_farmer ON alerts(farmer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_unread ON alerts(farmer_id, is_read)")
        
        conn.commit()
        print("Enhanced tables created successfully!")
    finally:
        conn.close()


# ============================================================
# 3. FARMER FUNCTIONS (CONTEXT-AWARE)
# ============================================================

def insert_farmer(name: str, state: str, phone: str = None, 
                  district: str = None, language: str = "English",
                  main_crop: str = None, farm_size: str = "Medium",
                  preferred_market: str = None, emergency_contact: str = None,
                  risk_preference: str = "Medium") -> int:
    """
    Insert a new farmer with full context.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO farmers (name, phone, state, district, language, 
                               main_crop, farm_size, preferred_market, 
                               emergency_contact, risk_preference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, phone, state, district, language, main_crop, 
              farm_size, preferred_market, emergency_contact, risk_preference))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_farmer(farmer_id: int) -> Optional[Dict]:
    """Fetch farmer with full context."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE id = ?", (farmer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_farmer_by_phone(phone: str) -> Optional[Dict]:
    """Fetch farmer by phone number for quick login."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE phone = ?", (phone,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_farmers() -> List[Dict]:
    """Fetch all farmers from database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_farmer(farmer_id: int, **kwargs) -> bool:
    """Update farmer information."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Build dynamic update query
        fields = []
        values = []
        for key, value in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(value)
        
        fields.append("updated_at = CURRENT_TIMESTAMP")
        
        query = f"UPDATE farmers SET {', '.join(fields)} WHERE id = ?"
        values.append(farmer_id)
        
        cursor.execute(query, values)
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================================
# 4. CROP FUNCTIONS (ENHANCED)
# ============================================================

def insert_crop(name: str, season: str = None, category: str = None,
                avg_growth_days: int = None, min_price: int = None,
                max_price: int = None) -> int:
    """Insert a new crop with market data."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO crops (name, season, category, avg_growth_days, 
                                       min_price_expected, max_price_expected)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, season, category, avg_growth_days, min_price, max_price))
        conn.commit()
        
        cursor.execute("SELECT id FROM crops WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_all_crops() -> List[Dict]:
    """Fetch all crops."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crops ORDER BY category, name")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_crops_by_season(season: str) -> List[Dict]:
    """Get crops by growing season."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crops WHERE season = ? ORDER BY name", (season,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def insert_multiple_crops(crops_list: List[Tuple]):
    """Insert multiple crops at once."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR IGNORE INTO crops (name, season, category, avg_growth_days, 
                                       min_price_expected, max_price_expected)
            VALUES (?, ?, ?, ?, ?, ?)
        """, crops_list)
        conn.commit()
    finally:
        conn.close()


# ============================================================
# 5. MARKET FUNCTIONS (ENHANCED)
# ============================================================

def insert_market(name: str, state: str, district: str,
                  latitude: float = None, longitude: float = None) -> int:
    """Insert a new market."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO markets (name, state, district, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
        """, (name, state, district, latitude, longitude))
        conn.commit()
        
        cursor.execute("SELECT id FROM markets WHERE name = ? AND state = ? AND district = ?", 
                      (name, state, district))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_markets_by_state_district(state: str, district: str = None) -> List[Dict]:
    """Get markets by state and optionally district."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if district:
            cursor.execute("""
                SELECT * FROM markets WHERE state = ? AND district = ? 
                AND is_active = 1 ORDER BY name
            """, (state, district))
        else:
            cursor.execute("""
                SELECT * FROM markets WHERE state = ? AND is_active = 1 
                ORDER BY name
            """, (state,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_best_and_nearby_market(state: str, crop_id: int, limit: int = 5) -> Dict:
    """
    Get best price market and nearby alternatives.
    This is the CORE function for "Where to sell?" decision.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Get highest price market (Best)
        cursor.execute("""
            SELECT m.*, p.modal_price, p.date
            FROM markets m
            JOIN prices p ON m.id = p.market_id
            WHERE m.state = ? AND p.crop_id = ?
            ORDER BY p.modal_price DESC
            LIMIT 1
        """, (state, crop_id))
        best = cursor.fetchone()
        
        # Get top 3 markets by price (for comparison)
        cursor.execute("""
            SELECT m.name, m.district, p.modal_price
            FROM markets m
            JOIN prices p ON m.id = p.market_id
            WHERE m.state = ? AND p.crop_id = ?
            ORDER BY p.modal_price DESC
            LIMIT ?
        """, (state, crop_id, limit))
        top_markets = [dict(row) for row in cursor.fetchall()]
        
        return {
            'best_market': dict(best) if best else None,
            'top_markets': top_markets
        }
    finally:
        conn.close()


# ============================================================
# 6. PRICE FUNCTIONS (CORE - TIME INTELLIGENCE)
# ============================================================

def insert_price(crop_id: int, market_id: int, date: str, 
                modal_price: float, min_price: float = None, 
                max_price: float = None, arrivals: int = None) -> int:
    """Insert price data with full details."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO prices 
            (crop_id, market_id, date, modal_price, min_price, max_price, arrivals)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (crop_id, market_id, date, modal_price, min_price, max_price, arrivals))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_price_by_crop_state(crop_name: str, state: str, limit: int = 30) -> pd.DataFrame:
    """
    Fetch price data based on crop and state.
    MAIN query for price forecasting.
    """
    conn = get_connection()
    try:
        query = """
            SELECT p.date, p.modal_price, p.min_price, p.max_price, p.arrivals,
                   m.name as market, m.district
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
            ORDER BY p.date DESC
            LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(crop_name, state, limit))
        return df
    finally:
        conn.close()


def get_price_history(crop_name: str, state: str, days: int = 30) -> pd.DataFrame:
    """Get price history for trend analysis."""
    conn = get_connection()
    try:
        query = """
            SELECT p.date, p.modal_price, p.min_price, p.max_price
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
            ORDER BY p.date ASC
            LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(crop_name, state, days))
        return df
    finally:
        conn.close()


def get_price_statistics(crop_name: str, state: str) -> Dict:
    """Get price statistics for decision making."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Latest price
        cursor.execute("""
            SELECT p.modal_price, p.date, m.name as market
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
            ORDER BY p.date DESC
            LIMIT 1
        """, (crop_name, state))
        latest = cursor.fetchone()
        
        # Average price (last 30 days)
        cursor.execute("""
            SELECT AVG(p.modal_price) as avg_price,
                   MIN(p.modal_price) as min_price,
                   MAX(p.modal_price) as max_price
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
            AND p.date >= date('now', '-30 days')
        """, (crop_name, state))
        stats = cursor.fetchone()
        
        return {
            'latest_price': latest['modal_price'] if latest else None,
            'latest_date': latest['date'] if latest else None,
            'latest_market': latest['market'] if latest else None,
            'avg_price': stats['avg_price'] if stats else None,
            'min_price': stats['min_price'] if stats else None,
            'max_price': stats['max_price'] if stats else None
        }
    finally:
        conn.close()


def calculate_trend(crop_name: str, state: str, days: int = 7) -> str:
    """
    Calculate price trend: increasing/decreasing/stable.
    Core logic for decision making.
    """
    df = get_price_history(crop_name, state, days=days * 2)
    
    if df.empty or len(df) < 2:
        return "stable"
    
    df = df.sort_values('date')
    
    # Compare first half vs second half
    mid = len(df) // 2
    recent = df.tail(mid)['modal_price'].mean()
    older = df.head(mid)['modal_price'].mean()
    
    # 2% threshold for trend detection
    if recent > older * 1.02:
        return "increasing"
    elif recent < older * 0.98:
        return "decreasing"
    else:
        return "stable"


def calculate_confidence(crop_name: str, state: str, days: int = 30) -> str:
    """
    Calculate prediction confidence based on data quality.
    Returns: High/Medium/Low
    """
    df = get_price_by_crop_state(crop_name, state, limit=days)
    
    if df.empty:
        return "Low"
    
    # Calculate data completeness
    completeness = len(df) / days * 100
    
    # Calculate price stability (coefficient of variation)
    if df['modal_price'].mean() > 0:
        cv = df['modal_price'].std() / df['modal_price'].mean()
    else:
        cv = 1
    
    if completeness >= 80 and cv < 0.15:
        return "High"
    elif completeness >= 50 and cv < 0.25:
        return "Medium"
    else:
        return "Low"


# ============================================================
# 7. PREDICTION FUNCTIONS (ENHANCED)
# ============================================================

def insert_prediction(farmer_id: int, crop: str, state: str,
                     current_price: float, predicted_price: float,
                     trend: str, confidence: str, decision: str,
                     decision_reason: str, money_impact: str = None,
                     model_used: str = "Moving Average",
                     data_quality: float = 100.0) -> int:
    """
    Save a price prediction with full reasoning.
    Includes decision_reason for explainability.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (farmer_id, crop, state, current_price, predicted_price, trend,
             confidence, decision, decision_reason, money_impact, 
             model_used, data_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (farmer_id, crop, state, current_price, predicted_price, trend,
              confidence, decision, decision_reason, money_impact,
              model_used, data_quality))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_predictions_by_farmer(farmer_id: int, limit: int = 10) -> List[Dict]:
    """Get prediction history for a farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE farmer_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (farmer_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_latest_prediction(crop: str, state: str) -> Optional[Dict]:
    """Get the most recent prediction for a crop/state."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE crop = ? AND state = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (crop, state))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ============================================================
# 8. ALERTS FUNCTIONS (NEW - PROACTIVE)
# ============================================================

def create_alert(farmer_id: int, alert_type: str, title: str, 
                message: str, severity: str = "Medium") -> int:
    """
    Create an alert for proactive notifications.
    Alert types: price_drop, weather_risk, disease, market, general
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (farmer_id, alert_type, title, message, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (farmer_id, alert_type, title, message, severity))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_alerts_for_farmer(farmer_id: int, unread_only: bool = False) -> List[Dict]:
    """Get alerts for a farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if unread_only:
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE farmer_id = ? AND is_read = 0
                ORDER BY severity DESC, created_at DESC
            """, (farmer_id,))
        else:
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE farmer_id = ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (farmer_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def mark_alert_read(alert_id: int) -> bool:
    """Mark an alert as read."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE alerts SET is_read = 1 WHERE id = ?", (alert_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_unread_alert_count(farmer_id: int) -> int:
    """Get count of unread alerts."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count FROM alerts 
            WHERE farmer_id = ? AND is_read = 0
        """, (farmer_id,))
        row = cursor.fetchone()
        return row['count'] if row else 0
    finally:
        conn.close()


# ============================================================
# 9. UTILITY FUNCTIONS
# ============================================================

def seed_initial_data():
    """Seed initial crops and markets data."""
    # Enhanced crops with market expectations
    crops = [
        ("Rice", "Kharif", "Cereal", 120, 1800, 2500),
        ("Wheat", "Rabi", "Cereal", 120, 2000, 2800),
        ("Cotton", "Kharif", "Fiber", 180, 5000, 7000),
        ("Sugarcane", "Annual", "Cash Crop", 365, 3000, 4500),
        ("Maize", "Kharif", "Cereal", 90, 1500, 2200),
        ("Tomato", "All Season", "Vegetable", 90, 1000, 3000),
        ("Potato", "Rabi", "Vegetable", 90, 1000, 2000),
        ("Onion", "Rabi", "Vegetable", 90, 1000, 2500),
        ("Groundnut", "Kharif", "Oilseed", 120, 4000, 6000),
        ("Mustard", "Rabi", "Oilseed", 120, 4500, 6500),
        ("Soybean", "Kharif", "Oilseed", 90, 3000, 5000),
        ("Barley", "Rabi", "Cereal", 100, 1500, 2200),
    ]
    insert_multiple_crops(crops)
    print("Enhanced initial data seeded!")


def get_database_stats() -> Dict:
    """Get database statistics."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        stats = {'status': 'success'}
        
        tables = ['farmers', 'crops', 'markets', 'prices', 'predictions', 'alerts']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()['count']
        
        return stats
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        conn.close()


def initialize_database():
    """Initialize database."""
    print("Initializing AgriDream database v2.0...")
    create_tables()
    seed_initial_data()
    stats = get_database_stats()
    print(f"Database ready: {stats}")
    return stats


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_database():
    """Test enhanced database operations."""
    print("Testing enhanced database...")
    
    # Initialize
    initialize_database()
    
    # Test farmer with context
    farmer_id = insert_farmer(
        name="Ramesh Patil",
        state="Maharashtra",
        phone="+919999999999",
        district="Pune",
        main_crop="Rice",
        farm_size="Large",
        preferred_market="Pune Market",
        emergency_contact="+918888888888"
    )
    print(f"Farmer inserted: ID {farmer_id}")
    
    # Test crop
    crop_id = insert_crop("Bajra", "Kharif", "Cereal", 90, 1200, 1800)
    print(f"Crop inserted: ID {crop_id}")
    
    # Test market
    market_id = insert_market("Pune APMC", "Maharashtra", "Pune")
    print(f"Market inserted: ID {market_id}")
    
    # Test price with date
    insert_price(crop_id, market_id, "2024-01-15", 1500.0, 1400.0, 1600.0, 100)
    print("Price inserted")
    
    # Test prediction with reason
    pred_id = insert_prediction(
        farmer_id=farmer_id,
        crop="Rice",
        state="Maharashtra",
        current_price=2200,
        predicted_price=2400,
        trend="increasing",
        confidence="High",
        decision="SELL NOW",
        decision_reason="Prices show upward trend over 7 days with high confidence",
        money_impact="+200 per quintal",
        model_used="Moving Average + Trend Analysis",
        data_quality=95.0
    )
    print(f"Prediction inserted: ID {pred_id}")
    
    # Test alert
    alert_id = create_alert(
        farmer_id=farmer_id,
        alert_type="price_drop",
        title="Price Alert: Rice",
        message="Rice prices have dropped by 5% this week in your region",
        severity="High"
    )
    print(f"Alert created: ID {alert_id}")
    
    # Test functions
    trend = calculate_trend("Rice", "Maharashtra")
    confidence = calculate_confidence("Rice", "Maharashtra")
    print(f"Trend: {trend}, Confidence: {confidence}")
    
    stats = get_database_stats()
    print(f"Database stats: {stats}")
    print("All enhanced tests passed!")


# ============================================================
# Chat History Functions for AgriCare AI
# ============================================================

def save_chat_message(farmer_id: int, user_message: str, bot_response: str, emotion: str = None, language: str = "English") -> int:
    """Save a chat message to the database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (farmer_id, user_message, bot_response, emotion, language)
            VALUES (?, ?, ?, ?, ?)
        """, (farmer_id, user_message, bot_response, emotion, language))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Error saving chat message: {e}")
        return -1
    finally:
        conn.close()


def get_chat_history(farmer_id: int, limit: int = 50) -> List[Dict]:
    """Retrieve chat history for a farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM chat_history 
            WHERE farmer_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (farmer_id, limit))
        
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "farmer_id": row["farmer_id"],
                "user": row["user_message"],
                "bot": row["bot_response"],
                "emotion": row["emotion"],
                "language": row["language"],
                "timestamp": row["created_at"]
            }
            for row in rows
        ]
    except sqlite3.Error as e:
        print(f"Error retrieving chat history: {e}")
        return []
    finally:
        conn.close()


def clear_chat_history(farmer_id: int) -> bool:
    """Clear chat history for a farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE farmer_id = ?", (farmer_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error clearing chat history: {e}")
        return False
    finally:
        conn.close()


def log_emergency_alert(farmer_id: int, message: str, contact_name: str, contact_phone: str) -> int:
    """Log emergency alert sent to family member."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (farmer_id, user_message, bot_response, emotion, language)
            VALUES (?, ?, ?, ?, ?)
        """, (farmer_id, f"EMERGENCY: {message}", f"Alert sent to {contact_name} ({contact_phone})", "high_risk", "system"))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Error logging emergency alert: {e}")
        return -1
    finally:
        conn.close()


if __name__ == "__main__":
    test_database()
