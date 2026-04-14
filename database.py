"""
AgriDream Database Module
=========================
Modular, beginner-friendly SQLite database layer for AgriDream app.
Optimized for read-heavy operations with proper error handling.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Database file path
DB_PATH = "agridream.db"


# ============================================================
# 1. CONNECTION MANAGEMENT
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Get SQLite database connection.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise


# ============================================================
# 2. TABLE CREATION
# ============================================================

def create_tables():
    """Create all required tables if they don't exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Farmers Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT,
                state TEXT,
                language TEXT DEFAULT 'English',
                emergency_contact TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Crops Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                season TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Markets Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, state)
            )
        """)
        
        # Prices Table (Read-heavy - optimized with indexes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER NOT NULL,
                market_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                modal_price REAL,
                min_price REAL,
                max_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crop_id) REFERENCES crops(id),
                FOREIGN KEY (market_id) REFERENCES markets(id),
                UNIQUE(crop_id, market_id, date)
            )
        """)
        
        # Create indexes for fast reads
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_crop ON prices(crop_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_crop_date ON prices(crop_id, date)")
        
        # Predictions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER,
                crop TEXT NOT NULL,
                state TEXT,
                predicted_price REAL,
                decision TEXT,
                confidence TEXT,
                money_impact TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_farmer ON predictions(farmer_id)")
        
        conn.commit()
        print("Tables created successfully!")
    finally:
        conn.close()


# ============================================================
# 3. FARMER FUNCTIONS
# ============================================================

def insert_farmer(name: str, phone: str = None, state: str = None, 
                  language: str = "English", emergency_contact: str = None) -> int:
    """Insert a new farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO farmers (name, phone, state, language, emergency_contact)
            VALUES (?, ?, ?, ?, ?)
        """, (name, phone, state, language, emergency_contact))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_farmer(farmer_id: int) -> Optional[Dict]:
    """Fetch farmer by ID."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE id = ?", (farmer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_farmers() -> List[Dict]:
    """Fetch all farmers."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ============================================================
# 4. CROP FUNCTIONS
# ============================================================

def insert_crop(name: str, season: str = None) -> int:
    """Insert a new crop."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO crops (name, season)
            VALUES (?, ?)
        """, (name, season))
        conn.commit()
        
        cursor.execute("SELECT id FROM crops WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_crop_id(crop_name: str) -> Optional[int]:
    """Get crop ID by name."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM crops WHERE name = ?", (crop_name,))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_all_crops() -> List[Dict]:
    """Fetch all crops."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crops ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def insert_multiple_crops(crops_list: List[Tuple[str, str]]):
    """Insert multiple crops at once."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR IGNORE INTO crops (name, season)
            VALUES (?, ?)
        """, crops_list)
        conn.commit()
    finally:
        conn.close()


# ============================================================
# 5. MARKET FUNCTIONS
# ============================================================

def insert_market(name: str, state: str, district: str = None) -> int:
    """Insert a new market."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO markets (name, state, district)
            VALUES (?, ?, ?)
        """, (name, state, district))
        conn.commit()
        
        cursor.execute("SELECT id FROM markets WHERE name = ? AND state = ?", (name, state))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_market_id(name: str, state: str) -> Optional[int]:
    """Get market ID by name and state."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM markets WHERE name = ? AND state = ?", (name, state))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()


def get_markets_by_state(state: str) -> List[Dict]:
    """Get all markets in a state."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markets WHERE state = ? ORDER BY name", (state,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_markets() -> List[Dict]:
    """Fetch all markets."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markets ORDER BY state, name")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ============================================================
# 6. PRICE FUNCTIONS (Read-Heavy Optimized)
# ============================================================

def insert_price(crop_id: int, market_id: int, date: str, 
                modal_price: float, min_price: float = None, 
                max_price: float = None) -> int:
    """Insert price data."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO prices (crop_id, market_id, date, modal_price, min_price, max_price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (crop_id, market_id, date, modal_price, min_price, max_price))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_price_by_crop_state(crop_name: str, state: str, limit: int = 30) -> pd.DataFrame:
    """
    Fetch price data based on crop and state.
    This is the MAIN query used in price forecasting.
    
    Args:
        crop_name: Name of the crop (e.g., "Rice")
        state: State name (e.g., "Maharashtra")
        limit: Number of recent records (default: 30)
    
    Returns:
        pd.DataFrame: Price data with columns [date, modal_price, min_price, max_price]
    """
    conn = get_connection()
    try:
        query = """
            SELECT p.date, p.modal_price, p.min_price, p.max_price
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


def get_latest_price(crop_name: str, state: str) -> Optional[Dict]:
    """Get the most recent price for a crop in a state."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.date, p.modal_price, p.min_price, p.max_price, m.name as market
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
            ORDER BY p.date DESC
            LIMIT 1
        """, (crop_name, state))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_average_price(crop_name: str, state: str) -> Optional[float]:
    """Get average price for crop in state."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(p.modal_price) as avg_price
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            WHERE c.name = ? AND m.state = ?
        """, (crop_name, state))
        row = cursor.fetchone()
        return row['avg_price'] if row else None
    finally:
        conn.close()


def get_price_trend(crop_name: str, state: str, days: int = 7) -> str:
    """Calculate price trend (increasing/decreasing/stable)."""
    df = get_price_by_crop_state(crop_name, state, limit=days * 2)
    
    if df.empty or len(df) < 2:
        return "stable"
    
    df = df.sort_values('date')
    recent = df.head(days // 2)['modal_price'].mean()
    older = df.tail(days // 2)['modal_price'].mean()
    
    if recent > older * 1.02:
        return "increasing"
    elif recent < older * 0.98:
        return "decreasing"
    else:
        return "stable"


def get_all_prices() -> pd.DataFrame:
    """Fetch all prices."""
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT c.name as crop, m.name as market, m.state, 
                   p.date, p.modal_price, p.min_price, p.max_price
            FROM prices p
            JOIN crops c ON p.crop_id = c.id
            JOIN markets m ON p.market_id = m.id
            ORDER BY p.date DESC
        """, conn)
    finally:
        conn.close()


# ============================================================
# 7. PREDICTION FUNCTIONS
# ============================================================

def insert_prediction(farmer_id: int, crop: str, state: str,
                     predicted_price: float, decision: str,
                     confidence: str, money_impact: str) -> int:
    """Save a price prediction result."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (farmer_id, crop, state, predicted_price, decision, confidence, money_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (farmer_id, crop, state, predicted_price, decision, confidence, money_impact))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_predictions_by_farmer(farmer_id: int) -> List[Dict]:
    """Get all predictions for a farmer."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE farmer_id = ?
            ORDER BY created_at DESC
        """, (farmer_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_recent_predictions(limit: int = 10) -> List[Dict]:
    """Get most recent predictions."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ============================================================
# 8. UTILITY FUNCTIONS
# ============================================================

def seed_initial_data():
    """Seed initial crops data."""
    crops = [
        ("Rice", "Kharif"),
        ("Wheat", "Rabi"),
        ("Cotton", "Kharif"),
        ("Sugarcane", "Annual"),
        ("Maize", "Kharif"),
        ("Tomato", "All Season"),
        ("Potato", "Rabi"),
        ("Onion", "Rabi"),
        ("Groundnut", "Kharif"),
        ("Mustard", "Rabi"),
        ("Soybean", "Kharif"),
        ("Barley", "Rabi"),
    ]
    insert_multiple_crops(crops)
    print("Initial data seeded!")


def get_database_stats() -> Dict:
    """Get database statistics."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM farmers")
        stats['farmers'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM crops")
        stats['crops'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM markets")
        stats['markets'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM prices")
        stats['prices'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        stats['predictions'] = cursor.fetchone()['count']
        
        return stats
    finally:
        conn.close()


def initialize_database():
    """Initialize database."""
    print("Initializing AgriDream database...")
    create_tables()
    seed_initial_data()
    stats = get_database_stats()
    print(f"Database ready: {stats}")
    return stats


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_database():
    """Test database operations."""
    print("Testing database...")
    
    # Initialize database first
    initialize_database()
    
    # Test connection
    conn = get_connection()
    print("Connection: OK")
    conn.close()
    
    # Test farmer insert
    farmer_id = insert_farmer("Test Farmer", "+919999999999", "Maharashtra", "English", "+918888888888")
    print(f"Farmer inserted: ID {farmer_id}")
    
    # Test crop insert
    crop_id = insert_crop("TestCrop", "Kharif")
    print(f"Crop inserted: ID {crop_id}")
    
    # Test market insert
    market_id = insert_market("Test Market", "Maharashtra", "Test District")
    print(f"Market inserted: ID {market_id}")
    
    # Test price insert
    price_id = insert_price(crop_id, market_id, "2024-01-01", 1000.0, 900.0, 1100.0)
    print(f"Price inserted: ID {price_id}")
    
    # Test prediction insert
    pred_id = insert_prediction(farmer_id, "TestCrop", "Maharashtra", 1100.0, "HOLD", "Medium", "+100")
    print(f"Prediction inserted: ID {pred_id}")
    
    # Test fetch
    df = get_price_by_crop_state("TestCrop", "Maharashtra")
    print(f"Price data fetched: {len(df)} rows")
    
    # Test stats
    stats = get_database_stats()
    print(f"Database stats: {stats}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_database()
