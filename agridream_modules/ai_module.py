"""
AgriCare AI Module - Handles all AI/ML responses
"""
import cohere
import random
import hashlib

# Cohere API Key
COHERE_API_KEY = '6rzDJkHIdCEw1aoURMqEqAk5kEZmTNDvXS7dQHbP'

def get_cohere_response(user_message, emotion, lang, farmer_profile=None, conversation_history=None):
    """Generate response using Cohere API"""
    try:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
        
        system_prompt = f"""
        You are AgriCare AI, a friendly and supportive assistant for farmers.
        Your role is to answer farming questions in simple, practical language and also provide emotional support.
        If the farmer sounds stressed, sad, or in a serious emotional state, first reply with empathy and comforting words.
        Then, clearly suggest sending an emergency alert message to their family or agricultural officer via WhatsApp/SMS.
        
        When giving farming advice:
        - Be specific and practical (fertilizers, irrigation methods, pest control, weather tips, crop care).
        - Keep answers short, clear, and positive.
        - Never give harmful or unsafe instructions.
        Use a warm, motivating tone — like a trusted friend.
        You may reply in English, or mix English with Tamil if that makes the farmer more comfortable.
        
        Current farmer: {farmer_name}
        Detected emotion: {emotion}
        Language preference: {lang}
        """
        
        co = cohere.Client(api_key=COHERE_API_KEY)
        
        chat_history = []
        if conversation_history:
            recent_chats = conversation_history[-3:]
            for chat in recent_chats:
                if chat.get('user') and chat.get('bot'):
                    chat_history.append({"role": "USER", "message": chat['user']})
                    chat_history.append({"role": "CHATBOT", "message": chat['bot']})
        
        chat_history.append({"role": "USER", "message": user_message})
        
        response = co.chat(
            model="command-a-03-2025",
            preamble=system_prompt,
            chat_history=chat_history,
            message=user_message,
            temperature=0.8,
            max_tokens=300
        )
        
        return response.text
        
    except Exception as e:
        print(f"Cohere API Error: {e}")
        return get_fallback_response(user_message, emotion)


def get_fallback_response(user_message, emotion):
    """Enhanced fallback responses when API fails"""
    import hashlib
    msg_hash = int(hashlib.md5(str(user_message).encode()).hexdigest()[:8], 16) if user_message else 0
    
    user_msg_lower = user_message.lower() if user_message else ""
    
    # Check query type
    farming_keywords = {
        'price': ['price', 'cost', 'rate', 'market', 'sell', 'buy', 'rupee', 'income'],
        'weather': ['weather', 'rain', 'monsoon', 'drought', 'temperature', 'humidity'],
        'crop': ['crop', 'plant', 'grow', 'harvest', 'field', 'paddy', 'wheat', 'rice'],
        'disease': ['disease', 'pest', 'insect', 'fungal', 'virus', 'sick'],
        'soil': ['soil', 'fertilizer', 'nutrient', 'nitrogen', 'phosphorus', 'potassium'],
        'irrigation': ['water', 'irrigation', 'drip', 'sprinkler', 'canal']
    }
    
    query_type = None
    for qtype, keywords in farming_keywords.items():
        if any(kw in user_msg_lower for kw in keywords):
            query_type = qtype
            break
    
    # Price responses
    if query_type == 'price':
        prices = [
            "💰 Market Prices: Tomato ₹18-25/kg, Potato ₹15-20/kg, Onion ₹20-30/kg, Rice ₹2100-2300/q, Wheat ₹2150-2400/q. Check Price page for more!",
            "📊 Current Rates: Tomato ₹18-25, Potato ₹15-20, Onion ₹20-30, Paddy ₹2100-2300, Wheat ₹2150-2400/q.",
            "💵 Today's Prices: Vegetable prices ₹15-30/kg, Grains ₹2100-2400/q. Mandi rates vary - check local market!"
        ]
        return prices[msg_hash % len(prices)]
    
    # Weather responses
    elif query_type == 'weather':
        weathers = [
            "🌤️ Weather: 28-35°C, Humidity 60-70%, Rain in 3-5 days. Good for Kharif! Delay irrigation, protect seedlings.",
            "🌧️ Monsoon Update: Light rain next 5 days, 25-32°C. Good for rice sowing. Avoid spraying before rain.",
            "🌦️ Forecast: Days 1-3 light rain, 4-7 clear. Great for planting rice & soybean!"
        ]
        return weathers[msg_hash % len(weathers)]
    
    # Crop responses
    elif query_type == 'crop':
        crops = [
            "🌾 Kharif Crops: 1)Rice 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Tips: certified seeds, soil test, monitor pests.",
            "🌱 Best Crops: 1)Paddy 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Use resistant varieties, rotate crops!",
            "🌿 This Season: Rice, Soybean, Cotton ideal now. Tips: organic compost, NPK based on soil test!"
        ]
        return crops[msg_hash % len(crops)]
    
    # Disease responses
    elif query_type == 'disease':
        diseases = [
            "🩺 Diseases: Vegetables-Blight use copper fungicide, Rice-Blight use resistant varieties. Prevention: disease-free seeds, crop rotation.",
            "🦠 Common Issues: Leaf spots-copper fungicide, Fruit rot-drainage+mulch. Prevention: spacing, remove debris. Use Disease Detection!"
        ]
        return diseases[msg_hash % len(diseases)]
    
    # Soil responses
    elif query_type == 'soil':
        return """🧪 Soil Health Tips

Key Nutrients for Crops:
• Nitrogen (N): For leafy growth - green color
• Phosphorus (P): For root & flower development
• Potassium (K): For disease resistance & fruit quality

Recommended pH Level: 6.0-7.5

💡 Tip: Get your soil tested at local agricultural office!"""
    
    # Irrigation responses
    elif query_type == 'irrigation':
        return """💧 Irrigation Management

Water-Saving Techniques:
• Drip irrigation - 40-60% water savings
• Sprinkler system - uniform water distribution
• Mulching - reduces evaporation
• Early morning watering - less loss

💡 Tip: Irrigate at dawn for best results!"""
    
    # Default farming response
    default_responses = [
        "🌾 I'm here to help with farming! Ask me about crops, weather, prices, diseases, or soil care.",
        "👨‍🌾 As your farming assistant, I can help with crop selection, weather updates, market prices, and more!",
        "💚 Feel free to ask about any farming topic - I'm happy to help you grow better!"
    ]
    return default_responses[msg_hash % len(default_responses)]


def detect_emotion(text):
    """Detect emotion from text input"""
    text_lower = text.lower()
    
    # High risk keywords
    high_risk_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'no hope', 'better without me', 
                          'burden', 'worthless', 'depressed', 'hopeless', 'give up', 'ending everything',
                          'death', 'die', 'suicidal', 'end it all']
    
    for keyword in high_risk_keywords:
        if keyword in text_lower:
            return "high_risk"
    
    # Sad keywords
    sad_keywords = ['sad', 'upset', 'worry', 'worried', 'stress', 'stressed', 'problem', 'trouble', 
                   'loss', 'failed', 'bad', 'cry', 'crying', 'difficult', 'hard', 'alone']
    
    for keyword in sad_keywords:
        if keyword in text_lower:
            return "sad"
    
    # Angry keywords
    angry_keywords = ['angry', 'frustrated', 'hate', 'annoyed', 'irritated', 'furious', 'unfair']
    
    for keyword in angry_keywords:
        if keyword in text_lower:
            return "angry"
    
    # Happy keywords
    happy_keywords = ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'thank', 'thanks', 
                     'joy', 'excited', 'love', 'best', 'wonderful', 'pleased', 'grateful']
    
    for keyword in happy_keywords:
        if keyword in text_lower:
            return "happy"
    
    return "happy"  # Default to happy


def send_emergency_alert(farmer_name, phone, message):
    """Send emergency WhatsApp alert via CallMeBot"""
    try:
        import requests
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Get CallMeBot credentials from environment
        callmebot_api_key = os.getenv('CALLMEBOT_API_KEY')
        callmebot_phone = os.getenv('CALLMEBOT_PHONE')
        
        if not callmebot_api_key or not callmebot_phone:
            print("CallMeBot API not configured")
            return False
        
        # Format phone number properly (add country code if needed)
        phone = phone.strip().replace('+', '')
        if not phone.startswith('91') and len(phone) == 10:
            phone = '91' + phone
        
        # Use CallMeBot API to send WhatsApp
        encoded_message = requests.utils.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={callmebot_phone}&text={encoded_message}&apikey={callmebot_api_key}"
        
        response = requests.get(url, timeout=15)
        print(f"WhatsApp alert response: {response.status_code}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"WhatsApp alert error: {e}")
        return False