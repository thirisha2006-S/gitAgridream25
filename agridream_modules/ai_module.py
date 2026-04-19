"""
AgriCare AI Module - Handles all AI/ML responses
"""
import cohere
import random
import hashlib
import time

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
            model="command-r-08-2024",
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
    """Enhanced fallback responses when API fails - analyzes message for contextual response"""
    import hashlib
    import time
    
    msg_hash = int(hashlib.md5(str(user_message).encode()).hexdigest()[:8], 16) if user_message else 0
    current_time_ms = int(time.time() * 1000)
    
    user_msg_lower = user_message.lower() if user_message else ""
    
    # FIRST: Check for emotional/mental health crisis patterns (highest priority)
    crisis_patterns = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'no hope', 'better without me',
        'burden', 'worthless', 'depressed', 'hopeless', 'give up', 'ending everything',
        'death', 'die', 'suicidal', 'end it all', 'going to die', 'kill me', 'not worth',
        'nothing matters', 'no reason', 'give up on me', 'end it', 'final'
    ]
    if any(p in user_msg_lower for p in crisis_patterns):
        crisis_responses = [
            "I hear you, and I care about you. You're not alone. Please reach out to someone you trust - a family member, friend, or helpline. Your life matters. Can we talk about what's making you feel this way?",
            "I'm concerned about you. Please remember that help is available. You can call a helpline or talk to someone you trust. You're important and your feelings are valid. Let's talk about what's happening.",
            "I understand you're going through a difficult time. Please reach out for support - talk to a family member, friend, or call a helpline. You don't have to face this alone. I'm here to listen."
        ]
        return crisis_responses[current_time_ms % len(crisis_responses)]
    
    # SECOND: Check for sad/down patterns
    sad_patterns = ['feeling bad', 'feeling sad', 'feel bad', 'feel sad', 'not feeling good', 
                   'feeling down', 'feeling low', 'upset', 'worried', 'stress', 'stressed',
                   'feeling terrible', 'not okay', 'sad', 'depressed', 'cry', 'crying',
                   'alone', 'tired', 'exhausted', 'lost', 'hopeless', 'help']
    if any(p in user_msg_lower for p in sad_patterns):
        sad_responses = [
            "I'm here for you. I can hear you're going through a tough time. Would you like to talk about what's troubling you? Remember, it's okay to not be okay.",
            "I understand you're feeling down. That's completely valid. I'm here to listen. What's been on your mind?",
            "I'm sorry you're feeling this way. You're not alone - I'm here with you. Would you like to share what's making you feel bad?"
        ]
        return sad_responses[current_time_ms % len(sad_responses)]
    
    # THIRD: Check for acknowledgment patterns (ok, yeah, yes, etc.)
    ack_patterns = ['ok', 'okay', 'yeah', 'yes', 'sure', 'alright', 'fine', 'good']
    if any(p in user_msg_lower for p in ack_patterns):
        ack_responses = [
            "Great! How can I help you today? Ask me about weather, crops, prices, or anything else!",
            "Awesome! What would you like to know about farming?",
            "Perfect! I'm here to help. What do you need assistance with?"
        ]
        return ack_responses[current_time_ms % len(ack_responses)]
    
    # FOURTH: Check for emotional support requests
    support_patterns = ['talk', 'share', 'chat', 'listen', 'help me', 'need someone']
    if any(p in user_msg_lower for p in support_patterns):
        support_responses = [
            "Of course! I'm here to listen. What's on your mind?",
            "I'm here for you. Tell me what's going on.",
            "Absolutely! I'm here to chat. What's been happening?"
        ]
        return support_responses[current_time_ms % len(support_responses)]
    
    # Analyze the message type and respond contextually
    # Greeting patterns
    greeting_patterns = ['hi', 'hello', 'hey', 'namaste', 'vanakkam', 'salam', 'good morning', 'good evening', 'good afternoon', 'hii', 'helo', 'hallo']
    if any(greet in user_msg_lower for greet in greeting_patterns):
        greetings = [
            f"Namaste! 🌾 How can I help you today?",
            f"Hello! 🌱 How are you doing? Is there something I can help you with?",
            f"Hey there! 🚜 Welcome to AgriCare AI! What would you like to know?",
            f"Hi! 💚 Great to hear from you! Ask me about farming, weather, prices, or just chat!",
            f"Namaste, friend! 🌻 How can I assist you today?"
        ]
        return greetings[current_time_ms % len(greetings)]
    
    # How are you patterns
    how_patterns = ['how are you', 'how r u', 'howdy', 'kaise ho', 'evaru', 'status', 'you doing']
    if any(p in user_msg_lower for p in how_patterns):
        how_responses = [
            "I'm doing well, thank you for asking! 🌾 Ready to help you with your farming questions.",
            "I'm here and ready to help! 💚 How can I assist you today?",
            "Doing great! 🚜 Looking forward to answering your questions about crops, weather, or anything else!"
        ]
        return how_responses[current_time_ms % len(how_responses)]
    
    # What can you do patterns
    help_patterns = ['what can you do', 'help me', 'what can you help', 'features', 'your work']
    if any(p in user_msg_lower for p in help_patterns):
        return """I can help you with:

Weather - Get forecast and monsoon updates
Prices - Check current market rates
Crops - Get recommendations for your land
Diseases - Identify plant health issues
Irrigation - Water-saving tips
Soil - Soil health and fertilizer advice
Chat - Just talk or share how you're feeling!

What would you like to know about?"""
    
    # Thank you patterns
    thank_patterns = ['thank', 'thanks', 'thx', 'appreciate', 'grateful']
    if any(p in user_msg_lower for p in thank_patterns):
        thanks_responses = [
            "You're welcome! 😊 Happy to help! Anything else?",
            "No problem! 🌾 Feel free to ask anytime!",
            "Glad I could help! 💚 What else can I do for you?"
        ]
        return thanks_responses[current_time_ms % len(thanks_responses)]
    
    # Goodbye patterns
    bye_patterns = ['bye', 'goodbye', 'see you', 'take care', 'valhalla']
    if any(p in user_msg_lower for p in bye_patterns):
        bye_responses = [
            "Goodbye, friend! 🌾 Take care of your crops!",
            "Namaste! 🚜 Hope to see you again soon!",
            "Bye! 💚 Wishing you a great harvest!"
        ]
        return bye_responses[current_time_ms % len(bye_responses)]
    
    # Check for farming-related keywords
    farming_keywords = {
        'price': ['price', 'cost', 'rate', 'market', 'sell', 'buy', 'rupee', 'income', 'bhaw', 'भाव'],
        'weather': ['weather', 'rain', 'monsoon', 'drought', 'temperature', 'humidity', 'forecast', 'rain'],
        'crop': ['crop', 'plant', 'grow', 'harvest', 'field', 'paddy', 'wheat', 'rice', 'cotton', 'sugarcane'],
        'disease': ['disease', 'pest', 'insect', 'fungal', 'virus', 'sick', 'yellow', 'drying'],
        'soil': ['soil', 'fertilizer', 'nutrient', 'nitrogen', 'phosphorus', 'potassium', 'ph'],
        'irrigation': ['water', 'irrigation', 'drip', 'sprinkler', 'canal', 'borewell']
    }
    
    query_type = None
    for qtype, keywords in farming_keywords.items():
        if any(kw in user_msg_lower for kw in keywords):
            query_type = qtype
            break
    
    # If it's a farming query, provide farming-specific response
    if query_type == 'price':
        prices = [
            "Market Prices: Tomato 18-25/kg, Potato 15-20/kg, Onion 20-30/kg, Rice 2100-2300/q, Wheat 2150-2400/q. Check Price page for more!",
            "Current Rates: Tomato 18-25, Potato 15-20, Onion 20-30, Paddy 2100-2300, Wheat 2150-2400/q.",
            "Today's Prices: Vegetable prices 15-30/kg, Grains 2100-2400/q. Mandi rates vary - check local market!"
        ]
        return prices[current_time_ms % len(prices)]
    
    elif query_type == 'weather':
        weathers = [
            "Weather: 28-35C, Humidity 60-70%, Rain in 3-5 days. Good for Kharif! Delay irrigation, protect seedlings.",
            "Monsoon Update: Light rain next 5 days, 25-32C. Good for rice sowing. Avoid spraying before rain.",
            "Forecast: Days 1-3 light rain, 4-7 clear. Great for planting rice and soybean!"
        ]
        return weathers[current_time_ms % len(weathers)]
    
    elif query_type == 'crop':
        crops = [
            "Kharif Crops: 1)Rice 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Tips: certified seeds, soil test, monitor pests.",
            "Best Crops: 1)Paddy 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Use resistant varieties, rotate crops!",
            "This Season: Rice, Soybean, Cotton ideal now. Tips: organic compost, NPK based on soil test!"
        ]
        return crops[current_time_ms % len(crops)]
    
    elif query_type == 'disease':
        diseases = [
            "Diseases: Vegetables-Blight use copper fungicide, Rice-Blight use resistant varieties. Prevention: disease-free seeds, crop rotation.",
            "Common Issues: Leaf spots-copper fungicide, Fruit rot-drainage+mulch. Prevention: spacing, remove debris. Use Disease Detection!"
        ]
        return diseases[current_time_ms % len(diseases)]
    
    elif query_type == 'soil':
        return """Soil Health Tips

Key Nutrients for Crops:
- Nitrogen (N): For leafy growth - green color
- Phosphorus (P): For root and flower development
- Potassium (K): For disease resistance and fruit quality

Recommended pH Level: 6.0-7.5

Tip: Get your soil tested at local agricultural office!"""
    
    elif query_type == 'irrigation':
        return """Irrigation Management

Water-Saving Techniques:
- Drip irrigation - 40-60% water savings
- Sprinkler system - uniform water distribution
- Mulching - reduces evaporation
- Early morning watering - less loss

Tip: Irrigate at dawn for best results!"""
    
    # For unrecognized messages - ask clarifying question
    # If message is short/random, ask clarifying question
    if len(user_message.strip()) < 3:
        clarify_responses = [
            "I see! Tell me more about what you'd like to know - I'm here to help!",
            "That's interesting! Would you like to ask about farming, weather, prices, or something else?",
            "I understand! How can I help you specifically? Ask me about crops, diseases, soil, or irrigation!",
            "Got it! What farming topic can I help you with today?",
            "I'd love to help! Ask me about weather, crops, prices, or just share how you're feeling."
        ]
        return clarify_responses[current_time_ms % len(clarify_responses)]
    
    # For any other message, try to be more helpful
    return f"""I understand you said: "{user_message}"

I can help you with:
- Weather and forecasts
- Market prices
- Crop recommendations
- Plant diseases
- Irrigation tips
- Soil health

Or if you're feeling down, I'm here to listen and chat!

What would you like to know more about?"""
    
    query_type = None
    for qtype, keywords in farming_keywords.items():
        if any(kw in user_msg_lower for kw in keywords):
            query_type = qtype
            break
    
    # If it's a farming query, provide farming-specific response
    if query_type == 'price':
        prices = [
            "💰 Market Prices: Tomato ₹18-25/kg, Potato ₹15-20/kg, Onion ₹20-30/kg, Rice ₹2100-2300/q, Wheat ₹2150-2400/q. Check Price page for more!",
            "📊 Current Rates: Tomato ₹18-25, Potato ₹15-20, Onion ₹20-30, Paddy ₹2100-2300, Wheat ₹2150-2400/q.",
            "💵 Today's Prices: Vegetable prices ₹15-30/kg, Grains ₹2100-2400/q. Mandi rates vary - check local market!"
        ]
        return prices[current_time_ms % len(prices)]
    
    elif query_type == 'weather':
        weathers = [
            "🌤️ Weather: 28-35°C, Humidity 60-70%, Rain in 3-5 days. Good for Kharif! Delay irrigation, protect seedlings.",
            "🌧️ Monsoon Update: Light rain next 5 days, 25-32°C. Good for rice sowing. Avoid spraying before rain.",
            "🌦️ Forecast: Days 1-3 light rain, 4-7 clear. Great for planting rice & soybean!"
        ]
        return weathers[current_time_ms % len(weathers)]
    
    elif query_type == 'crop':
        crops = [
            "🌾 Kharif Crops: 1)Rice 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Tips: certified seeds, soil test, monitor pests.",
            "🌱 Best Crops: 1)Paddy 2)Soybean 3)Cotton 4)Sugarcane 5)Vegetables. Use resistant varieties, rotate crops!",
            "🌿 This Season: Rice, Soybean, Cotton ideal now. Tips: organic compost, NPK based on soil test!"
        ]
        return crops[current_time_ms % len(crops)]
    
    elif query_type == 'disease':
        diseases = [
            "🩺 Diseases: Vegetables-Blight use copper fungicide, Rice-Blight use resistant varieties. Prevention: disease-free seeds, crop rotation.",
            "🦠 Common Issues: Leaf spots-copper fungicide, Fruit rot-drainage+mulch. Prevention: spacing, remove debris. Use Disease Detection!"
        ]
        return diseases[current_time_ms % len(diseases)]
    
    elif query_type == 'soil':
        return """🧪 Soil Health Tips

Key Nutrients for Crops:
• Nitrogen (N): For leafy growth - green color
• Phosphorus (P): For root & flower development
• Potassium (K): For disease resistance & fruit quality

Recommended pH Level: 6.0-7.5

💡 Tip: Get your soil tested at local agricultural office!"""
    
    elif query_type == 'irrigation':
        return """💧 Irrigation Management

Water-Saving Techniques:
• Drip irrigation - 40-60% water savings
• Sprinkler system - uniform water distribution
• Mulching - reduces evaporation
• Early morning watering - less loss

💡 Tip: Irrigate at dawn for best results!"""
    
    # For unrecognized messages - ask clarifying question
    clarify_responses = [
        f"I see! Tell me more about what you'd like to know - I'm here to help! 🌾",
        f"That's interesting! 💚 Would you like to ask about farming, weather, prices, or something else?",
        f"I understand! 🌱 How can I help you specifically? Ask me about crops, diseases, soil, or irrigation!",
        f"Got it! 🚜 What farming topic can I help you with today?",
        f"I'd love to help! 💧 Ask me about weather, crops, prices, or just share how you're feeling."
    ]
    
    # If message is just random characters, still respond helpfully
    if len(user_message.strip()) < 3:
        return clarify_responses[current_time_ms % len(clarify_responses)]
    
    # For any other message - try to be helpful
    return f"""I understand you said: "{user_message}" 🌾

I can help you with:
• 🌤️ Weather & forecasts
• 💰 Market prices
• 🌱 Crop recommendations  
• 🩺 Plant diseases
• 💧 Irrigation tips
• 🧪 Soil health

What would you like to know more about?"""


def detect_emotion(text):
    """Detect emotion from text input"""
    text_lower = text.lower()
    
    # High risk keywords - more flexible matching
    high_risk_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'no hope', 'better without me', 
                          'burden', 'worthless', 'depressed', 'hopeless', 'give up', 'ending everything',
                          'death', 'die', 'suicidal', 'end it all', 'going to die', 'kill me', 'not worth']
    
    for keyword in high_risk_keywords:
        if keyword in text_lower:
            return "high_risk"
    
    # Sad keywords
    sad_keywords = ['sad', 'upset', 'worry', 'worried', 'stress', 'stressed', 'problem', 'trouble', 
                   'loss', 'failed', 'bad', 'cry', 'crying', 'difficult', 'hard', 'alone', 'tired', 
                   'depressed', 'feeling down', 'not good', 'worst']
    
    for keyword in sad_keywords:
        if keyword in text_lower:
            return "sad"
    
    # Angry keywords
    angry_keywords = ['angry', 'frustrated', 'hate', 'annoyed', 'irritated', 'furious', 'unfair', 'mad']
    
    for keyword in angry_keywords:
        if keyword in text_lower:
            return "angry"
    
    # Happy keywords
    happy_keywords = ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'thank', 'thanks', 
                     'joy', 'excited', 'love', 'best', 'wonderful', 'pleased', 'grateful', 'hi', 'hello', 
                     'bye', 'goodbye', 'nice', 'cool']
    
    for keyword in happy_keywords:
        if keyword in text_lower:
            return "happy"
    
    return "happy"  # Default to happy for unrecognized
    
    return "happy"  # Default to happy


def send_emergency_alert(farmer_name, phone, message):
    """Send emergency SMS alert via RapidAPI SMS service
    RapidAPI Key: User provided
    """
    try:
        import requests
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Get RapidAPI credentials from environment or Streamlit secrets
        rapidapi_key = os.getenv('RAPIDAPI_KEY')
        
        # Try Streamlit secrets as fallback
        try:
            import streamlit as st
            if 'RAPIDAPI_KEY' in st.secrets:
                rapidapi_key = st.secrets['RAPIDAPI_KEY']
        except:
            pass
        
        if not rapidapi_key:
            print("RapidAPI key not configured")
            # Fallback to Fast2SMS
            fast2sms_api_key = os.getenv('FAST2SMS_API_KEY')
            if fast2sms_api_key:
                return send_via_fast2sms(fast2sms_api_key, phone, message)
            return False
        
        # Format phone number (India - 10 digits)
        phone = phone.strip().replace('+', '').replace(' ', '')
        if len(phone) == 10:
            phone = '91' + phone
        elif len(phone) == 11 and phone.startswith('0'):
            phone = '91' + phone[1:]
        
        # Use RapidAPI - fast2sms service
        url = "https://fast2sms.p.rapidapi.com/send"

        payload = {
            "message": message,
            "language": "english",
            "route": "p",
            "numbers": phone
        }

        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": rapidapi_key,
            "X-RapidAPI-Host": "fast2sms.p.rapidapi.com"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        print(f"SMS alert response: {response.status_code} - {response.text}")
        
        # Check response properly
        try:
            resp_json = response.json()
            return resp_json.get('return', False)
        except:
            return response.status_code == 200
        
    except Exception as e:
        print(f"SMS alert error: {e}")
        return False


def send_via_fast2sms(api_key, phone, message):
    """Fallback: Send via Fast2SMS direct API"""
    try:
        import requests
        
        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = f"sender_id=FSTSMS&message={message}&language=english&route=p&numbers={phone}"
        headers = {'authorization': api_key, 'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(url, data=payload, headers=headers, timeout=15)
        return response.status_code == 200 and 'Success' in response.text
    except Exception as e:
        print(f"Fast2SMS error: {e}")
        return False