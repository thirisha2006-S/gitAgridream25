import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
# from googletrans import Translator  # Replaced with translate library for stability
from translate import Translator
import openai
import cohere
# Removed Twilio - using Deep AI instead
# Removed Hugging Face transformers - using Deep AI instead
TRANSFORMERS_AVAILABLE = False
print("Using Deep AI for enhanced features")



load_dotenv()

# ---------------------------
# Global Variables and Functions
# ---------------------------

# Safe translate function with error handling
def safe_translate(text, src='en', dest='en'):
    """Safely translate text with error handling"""
    try:
        if src == dest:
            return text
        translator_obj = Translator(from_lang=src, to_lang=dest)
        return translator_obj.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

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

# Crop recommendation function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, state):
    # Filter by state if needed, but for simplicity, use all data
    conditions = np.array([N, P, K, temperature, humidity, ph, rainfall])
    distances = np.sqrt(np.sum((df_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] - conditions) ** 2, axis=1))
    closest_indices = distances.nsmallest(3).index
    recommended_crops = df_crop.iloc[closest_indices]['label'].tolist()
    confidences = [max(0, min(100, 100 - distances[i] / 10)) for i in closest_indices]
    return recommended_crops, confidences

# Weather function
def get_weather(city):
    api_key = os.getenv('OPENWEATHER_API_KEY', '21e959d85a5148fdd18fbb293869d9ef')  # Demo key
    if not api_key or api_key == 'your_openweather_key_here':
        return None, "API key not configured"

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 200:
            weather = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'rainfall': data.get('rain', {}).get('1h', 0)  # Last hour rainfall
            }
            return weather, None
        else:
            return None, data.get('message', 'Weather data not available')
    except Exception as e:
        return None, str(e)

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
        "menu_emotion": "AgriCare AI",
        "menu_emergency": "Emergency Alert",
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

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

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
    farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

    # Get conversation context
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

    chatgpt_responses = {
        "happy": [
            f"That's wonderful to hear, {farmer_name}! 😊 You know, it's moments like these that make all the hard work worth it. What made today special for you?",
            f"I'm so glad you're feeling good, {farmer_name}! 🌟 Tell me more about what's bringing you joy right now.",
            f"That's fantastic! You deserve to feel this good, {farmer_name}. What's been the highlight of your week?"
        ],
        "sad": [
            f"Oh, {farmer_name}, I can really hear how heavy that feels right now. 🌱 You know, it's completely okay to have tough days. I'm right here with you. What specifically has been weighing on your mind?",
            f"I hear you, {farmer_name}. That sounds really difficult. Remember when we talked about {last_topic}? How are things going with that? I'm here to listen, no matter what.",
            f"That's so tough, {farmer_name}. I wish I could give you a big hug right now. 💙 What do you think might help you feel a little better today?"
        ],
        "angry": [
            f"I can feel how frustrated you are, {farmer_name}. 😠 That's completely understandable - farming can be incredibly challenging. What happened that made you feel this way?",
            f"Oh man, {farmer_name}, that sounds really frustrating! I totally get why you'd feel angry about that. You know what? You're absolutely right to feel this way. How can I support you through this?",
            f"That's so unfair, {farmer_name}. I can imagine how maddening that must be. 🌿 What do you think needs to change? I'm here to help you figure this out."
        ],
        "high_risk": [
            f"Oh, {farmer_name}, my heart goes out to you right now. 💔 I can hear how much pain you're in, and I want you to know you're not alone. Please remember how much you matter to the people who care about you. Can we talk about what might help you feel a little safer right now?",
            f"{farmer_name}, I hear the darkness in your words, and it breaks my heart. 🌙 You are so incredibly valuable, and there are people who love you deeply. Please reach out to someone you trust right now - I'm here with you, and help is available. What can I do to support you in this moment?",
            f"I feel your pain, {farmer_name}, and I want you to know how much I care about you. 💙 You're not alone in this darkness. Please talk to someone - a friend, family member, or helpline. You're stronger than you know, and there is hope. I'm right here with you."
        ]
    }

    responses = chatgpt_responses.get(emotion, [
        f"I hear you, {farmer_name}. You know, sometimes just talking about things can help. What's been on your mind lately?",
        f"That's interesting, {farmer_name}. Tell me more about that. I'm genuinely curious to hear your thoughts.",
        f"I appreciate you sharing that with me, {farmer_name}. How are you feeling about everything right now?"
    ])

    return responses[0]

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

    return responses[0]  # Return first response for consistency

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
    get_text("menu_emotion", global_lang)
]
menu = st.sidebar.radio(
    get_text("select_language", global_lang),
    menu_options
)


st.markdown(f"<h1 style='text-align:center; color:green;'>🌱 {get_text('title', global_lang)}</h1>", unsafe_allow_html=True)
st.markdown("---")

# Farmer Profile Setup
st.sidebar.subheader("👨‍🌾 " + get_text("farmer_profile", global_lang))
farmer_name = st.sidebar.text_input(get_text("farmer_name", global_lang), key="farmer_name")
farmer_age = st.sidebar.number_input(get_text("age", global_lang), min_value=18, max_value=100, value=30, key="farmer_age")
st.sidebar.subheader(get_text("emergency_contacts", global_lang))
family1_name = st.sidebar.text_input(get_text("family_member_1", global_lang), key="family1_name")
family1_phone = st.sidebar.text_input(get_text("phone", global_lang), key="family1_phone")
family2_name = st.sidebar.text_input(get_text("family_member_2", global_lang), key="family2_name")
family2_phone = st.sidebar.text_input(get_text("phone", global_lang), key="family2_phone")

if st.sidebar.button(get_text("save_profile", global_lang)):
    st.sidebar.success(get_text("profile_saved", global_lang))
    # Store in session state for emergency alerts
    st.session_state.farmer_profile = {
        "name": farmer_name,
        "age": farmer_age,
        "family1": {"name": family1_name, "phone": family1_phone},
        "family2": {"name": family2_name, "phone": family2_phone}
    }
    # Update global FAMILY_NUMBERS variable
    FAMILY_NUMBERS = get_family_numbers(st.session_state.farmer_profile)

# ---------------------------
# Dashboard
# ---------------------------
if menu == get_text("menu_dashboard", global_lang):
    st.subheader("💹 " + get_text("market_dashboard", global_lang))

    # Load price data
    df_prices = pd.read_csv('agmarknet_prices.csv')

    # Get top 10 commodities by modal price
    top_commodities = df_prices.nlargest(10, 'Modal_x0020_Price')[['Commodity', 'Modal_x0020_Price', 'State', 'Market', 'District']].drop_duplicates(subset=['Commodity'])

    st.write(f"### {get_text('top_commodities', global_lang)}")
    for i, (_, row) in enumerate(top_commodities.iterrows(), 1):
        st.write(f"{i}. **{row['Commodity']}** - ₹{int(row['Modal_x0020_Price'])} ({row['Market']}, {row['District']}, {row['State']})")

    st.write(f"### {get_text('market_insights', global_lang)}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text("total_commodities", global_lang), len(df_prices['Commodity'].unique()))
    with col2:
        st.metric(get_text("avg_price", global_lang), f"₹{int(df_prices['Modal_x0020_Price'].mean())}")
    with col3:
        st.metric(get_text("states_covered", global_lang), len(df_prices['State'].unique()))

    st.write(f"### {get_text('price_trends', global_lang)}")
    st.bar_chart(df_prices.groupby('Commodity')['Modal_x0020_Price'].mean().nlargest(10))

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
        recommended_crops, confidences = recommend_crop(N, P, K, temperature, humidity, ph, rainfall, state)

        st.write(f"### {get_text('top_3_crops', global_lang)}")
        for i, (crop, conf) in enumerate(zip(recommended_crops, confidences), 1):
            st.write(f"{i}. **{crop}** - {get_text('confidence', global_lang)}: {conf:.1f}%")

        # Irrigation Recommendation
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
            st.warning("Low rainfall detected. Consider increasing irrigation frequency or switching to more efficient methods.")
        elif 50 <= rainfall < 100:
            st.info("Moderate rainfall. Your irrigation method should complement natural rainfall effectively.")
        else:
            st.success("Good rainfall conditions. Your irrigation method will provide excellent backup during dry spells.")

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

    selected_state = st.selectbox(get_text("select_state", global_lang), available_states)

    # Filter crops by state and limit to 500
    if selected_state in df['State'].values:
        state_crops = df[df['State'] == selected_state]['Commodity'].unique()
        # Sort and limit to 500 crops
        available_crops = sorted(state_crops)[:500]
    else:
        # For states not in data, show comprehensive list of crops (expanded to ~500)
        available_crops = [
            # Cereals & Grains (50+)
            'Rice', 'Wheat', 'Maize', 'Barley', 'Oats', 'Bajra', 'Jowar', 'Ragi', 'Corn', 'Millet',
            'Quinoa', 'Buckwheat', 'Sorghum', 'Foxtail Millet', 'Little Millet', 'Kodo Millet', 'Barnyard Millet',
            'Brown Rice', 'Basmati Rice', 'Parboiled Rice', 'Red Rice', 'Black Rice', 'White Rice', 'Jasmine Rice',
            'Arborio Rice', 'Wild Rice', 'Glutinous Rice', 'Himalayan Red Rice', 'Purple Rice', 'Yellow Rice',
            'Durum Wheat', 'Bread Wheat', 'Spelt', 'Emmer', 'Einkorn', 'Kamut', 'Triticale', 'Rye',
            'Triticale', 'Amaranth', 'Teff', 'Fonio', 'Millet Mix', 'Popcorn', 'Flint Corn', 'Dent Corn',

            # Fruits (80+)
            'Banana', 'Apple', 'Orange', 'Mango', 'Grapes', 'Pineapple', 'Papaya', 'Pomegranate',
            'Guava', 'Lemon', 'Lime', 'Sweet Lime', 'Watermelon', 'Muskmelon', 'Strawberry', 'Kiwi',
            'Pear', 'Peach', 'Plum', 'Cherry', 'Apricot', 'Fig', 'Date', 'Coconut', 'Cashew',
            'Almond', 'Walnut', 'Pistachio', 'Raisin', 'Currant', 'Blueberry', 'Raspberry', 'Blackberry',
            'Cranberry', 'Gooseberry', 'Elderberry', 'Mulberry', 'Boysenberry', 'Loganberry', 'Tayberry',
            'Avocado', 'Dragon Fruit', 'Passion Fruit', 'Star Fruit', 'Jackfruit', 'Durian', 'Rambutan',
            'Lychee', 'Longan', 'Sapodilla', 'Tamarind', 'Custard Apple', 'Sugar Apple', 'Soursop',
            'Mangosteen', 'Salak', 'Langsat', 'Breadfruit', 'Plantain', 'Cooking Banana', 'Lady Finger Banana',
            'Red Banana', 'Cavendish Banana', 'Gros Michel Banana', 'Apple Fuji', 'Apple Gala', 'Apple Granny Smith',
            'Apple Honeycrisp', 'Apple Braeburn', 'Orange Navel', 'Orange Valencia', 'Orange Blood', 'Orange Cara Cara',
            'Mango Alphonso', 'Mango Kesar', 'Mango Dasheri', 'Mango Langra', 'Mango Chaunsa', 'Mango Totapuri',

            # Vegetables (100+)
            'Tomato', 'Potato', 'Onion', 'Carrot', 'Cabbage', 'Cauliflower', 'Brinjal', 'Capsicum',
            'Chilli', 'Garlic', 'Ginger', 'Turmeric', 'Radish', 'Turnip', 'Beetroot', 'Sweet Potato',
            'Yam', 'Taro', 'Cassava', 'Colocasia', 'Drumstick', 'Bitter Gourd', 'Bottle Gourd',
            'Ridge Gourd', 'Snake Gourd', 'Ash Gourd', 'Pointed Gourd', 'Sponge Gourd', 'Cucumber',
            'Pumpkin', 'Squash', 'Zucchini', 'Broccoli', 'Lettuce', 'Spinach', 'Fenugreek', 'Amaranth',
            'Palak', 'Methi', 'Coriander', 'Mint', 'Basil', 'Parsley', 'Celery', 'Fennel', 'Dill',
            'Thyme', 'Rosemary', 'Sage', 'Oregano', 'Tarragon', 'Chives', 'Leek', 'Scallion', 'Shallot',
            'Artichoke', 'Asparagus', 'Eggplant', 'Bell Pepper', 'Jalapeno', 'Habanero', 'Serrano',
            'Poblano', 'Anaheim', 'Banana Pepper', 'Cherry Tomato', 'Grape Tomato', 'Beefsteak Tomato',
            'Roma Tomato', 'Heirloom Tomato', 'Yellow Tomato', 'Green Tomato', 'Purple Tomato',
            'Red Potato', 'Yellow Potato', 'Blue Potato', 'Fingerling Potato', 'Russet Potato',
            'Yukon Gold Potato', 'New Potato', 'Sweet Potato Orange', 'Sweet Potato White', 'Sweet Potato Purple',
            'Yam African', 'Yam Asian', 'Taro Hawaiian', 'Cassava Bitter', 'Cassava Sweet', 'Arrowroot',
            'Jerusalem Artichoke', 'Salsify', 'Skirret', 'Parsnip', 'Rutabaga', 'Kohlrabi', 'Brussels Sprouts',
            'Kale', 'Collard Greens', 'Mustard Greens', 'Turnip Greens', 'Beet Greens', 'Swiss Chard',
            'Radicchio', 'Endive', 'Escarole', 'Frisee', 'Arugula', 'Watercress', 'Nasturtium',
            'Purslane', 'Sorrel', 'Malabar Spinach', 'New Zealand Spinach', 'Orach', 'Lambsquarters',

            # Spices & Herbs (50+)
            'Cumin', 'Coriander', 'Mustard', 'Fenugreek', 'Fennel', 'Cinnamon', 'Clove', 'Cardamom',
            'Nutmeg', 'Mace', 'Black Pepper', 'Red Chilli', 'Green Chilli', 'Bay Leaf', 'Thyme',
            'Rosemary', 'Sage', 'Oregano', 'Tarragon', 'Dill', 'Cilantro', 'Curry Leaf', 'Lemongrass',
            'Galangal', 'Turmeric', 'Ginger', 'Garlic', 'Shallot', 'Onion', 'Asafoetida', 'Saffron',
            'Vanilla', 'Star Anise', 'Sichuan Pepper', 'Cubeb', 'Grains of Paradise', 'Melegueta Pepper',
            'Allspice', 'Juniper Berry', 'Sumac', 'Za\'atar', 'Dukkah', 'Ras el Hanout', 'Garam Masala',
            'Curry Powder', 'Chili Powder', 'Paprika', 'Cayenne', 'Chipotle', 'Ancho', 'Guajillo',

            # Oilseeds & Nuts (40+)
            'Groundnut', 'Soybean', 'Sunflower', 'Sesame', 'Castor', 'Linseed', 'Flaxseed',
            'Mustard Seed', 'Rapeseed', 'Safflower', 'Cottonseed', 'Coconut Oil', 'Palm Oil',
            'Olive Oil', 'Almond', 'Walnut', 'Pistachio', 'Cashew', 'Peanut', 'Hazelnut',
            'Macadamia', 'Brazil Nut', 'Pecan', 'Chestnut', 'Beech Nut', 'Pine Nut', 'Chia Seed',
            'Hemp Seed', 'Pumpkin Seed', 'Sunflower Seed', 'Poppy Seed', 'Nigella Seed', 'Caraway',
            'Coriander Seed', 'Fennel Seed', 'Anise Seed', 'Cumin Seed', 'Fenugreek Seed',

            # Pulses & Legumes (50+)
            'Chickpea', 'Lentil', 'Pea', 'Kidney Bean', 'Black Gram', 'Green Gram', 'Pigeon Pea',
            'Horse Gram', 'Cowpea', 'Moth Bean', 'Urad', 'Moong', 'Rajma', 'Chana', 'Masoor',
            'Toor', 'Arhar', 'Kabuli Chana', 'Bengal Gram', 'Field Pea', 'Garden Pea', 'Snow Pea',
            'Sugar Snap Pea', 'Split Pea', 'Yellow Pea', 'Green Pea', 'Black-Eyed Pea', 'Lima Bean',
            'Fava Bean', 'Broad Bean', 'Runner Bean', 'Wax Bean', 'French Bean', 'String Bean',
            'Pinto Bean', 'Navy Bean', 'Great Northern Bean', 'Cannellini Bean', 'Red Kidney Bean',
            'Black Bean', 'Adzuki Bean', 'Mung Bean', 'Urad Bean', 'Moth Bean', 'Horse Gram',
            'Cowpea', 'Cluster Bean', 'Asparagus Bean', 'Yardlong Bean', 'Winged Bean',

            # Cash Crops & Fibers (30+)
            'Cotton', 'Jute', 'Sugarcane', 'Tobacco', 'Tea', 'Coffee', 'Rubber', 'Coconut',
            'Areca Nut', 'Betel Leaf', 'Opium Poppy', 'Coca', 'Quinine', 'Pyrethrum', 'Stevia',
            'Guar Gum', 'Cassia', 'Myrobalan', 'Wattle', 'Lac', 'Shellac', 'Resin', 'Gum Arabic',
            'Tragacanth', 'Karaya Gum', 'Guar Gum', 'Locust Bean Gum', 'Xanthan Gum', 'Gellan Gum',

            # Flowers & Ornamentals (30+)
            'Rose', 'Jasmine', 'Chrysanthemum', 'Marigold', 'Sunflower', 'Dahlia', 'Tulip',
            'Orchid', 'Carnation', 'Gerbera', 'Lily', 'Aster', 'Gladiolus', 'Daisy', 'Poppy',
            'Lavender', 'Hibiscus', 'Bougainvillea', 'Ixora', 'Allamanda', 'Oleander', 'Plumeria',
            'Gardenia', 'Night Jasmine', 'Rangoon Creeper', 'Madagascar Periwinkle', 'Adenium',
            'Kalanchoe', 'Sedum', 'Aloe Vera',

            # Medicinal Plants (40+)
            'Aloe Vera', 'Neem', 'Tulsi', 'Ashwagandha', 'Brahmi', 'Giloy', 'Amla', 'Haritaki',
            'Bibhitaki', 'Triphala', 'Sandalwood', 'Eucalyptus', 'Mint', 'Lemongrass', 'Ginger',
            'Turmeric', 'Garlic', 'Onion', 'Fenugreek', 'Cumin', 'Coriander', 'Fennel', 'Cardamom',
            'Cinnamon', 'Clove', 'Black Pepper', 'Long Pepper', 'Guggul', 'Boswellia', 'Myrrh',
            'Frankincense', 'Sandalwood', 'Agarwood', 'Patchouli', 'Vetiver', 'Lavender', 'Rosemary',
            'Thyme', 'Sage', 'Oregano', 'Basil', 'Holy Basil'
        ][:500]  # Limit to 500 as requested
        st.info(f"Price data for {selected_state} is coming soon. Showing comprehensive crop list.")

    crop_choice = st.selectbox(get_text("select_crop", global_lang), available_crops)

    # Get price data for selected crop and state
    crop_prices = df[(df['Commodity'] == crop_choice) & (df['State'] == selected_state)]
    if not crop_prices.empty:
        # Show current modal price
        current_price = crop_prices['Modal_x0020_Price'].iloc[0]
        st.metric(f"Current Modal Price for {crop_choice}", f"₹{current_price}")

        # Show price range
        min_price = crop_prices['Min_x0020_Price'].iloc[0]
        max_price = crop_prices['Max_x0020_Price'].iloc[0]
        st.write(f"**Price Range:** ₹{min_price} - ₹{max_price}")

        # Show market information
        market = crop_prices['Market'].iloc[0]
        district = crop_prices['District'].iloc[0]
        st.write(f"**Market:** {market}, {district}, {selected_state}")

        # Simple forecast (placeholder)
        st.markdown("### Price Forecast (Sample)")
        forecast_prices = [current_price * (1 + 0.05 * (i/30)) for i in range(30)]
        st.line_chart(forecast_prices)
    else:
        # For states not in data, show sample prices
        sample_prices = {
            'Rice': 2500, 'Wheat': 2200, 'Maize': 1800, 'Sugarcane': 3000, 'Cotton': 6000,
            'Tomato': 1500, 'Potato': 1200, 'Onion': 1000, 'Banana': 4000, 'Apple': 8000,
            'Orange': 3500, 'Mango': 5000, 'Grapes': 7000, 'Pineapple': 4500, 'Papaya': 3000,
            'Pomegranate': 6000, 'Carrot': 2000, 'Cabbage': 1800, 'Cauliflower': 2500,
            'Brinjal': 2200, 'Capsicum': 3000, 'Chilli': 4000, 'Garlic': 8000, 'Ginger': 10000,
            'Turmeric': 12000, 'Coriander': 5000, 'Cumin': 15000, 'Mustard': 6000,
            'Groundnut': 5500, 'Soybean': 4000, 'Sunflower': 4500, 'Barley': 1800, 'Oats': 2500,
            'Bajra': 1600, 'Jowar': 2000, 'Ragi': 2200, 'Chickpea': 4500, 'Lentil': 8000, 'Pea': 3000
        }
        current_price = sample_prices.get(crop_choice, 2000)  # Default price
        st.metric(f"Sample Modal Price for {crop_choice}", f"₹{current_price}")
        st.info("This is sample pricing data. Actual prices may vary by market and season.")
        st.write(f"**Sample Price Range:** ₹{int(current_price * 0.8)} - ₹{int(current_price * 1.2)}")
        st.write(f"**State:** {selected_state} (Sample Data)")

        # Simple forecast (placeholder)
        st.markdown("### Price Forecast (Sample)")
        forecast_prices = [current_price * (1 + 0.05 * (i/30)) for i in range(30)]
        st.line_chart(forecast_prices)

# ---------------------------
# Weather
# ---------------------------
elif menu == get_text("menu_weather", global_lang):
    st.subheader("🌤️ " + get_text("live_weather", global_lang))
    city = st.text_input(get_text("enter_city", global_lang), "Delhi")

    if st.button(get_text("get_weather", global_lang)):
        weather, error = get_weather(city)
        if weather:
            st.write(f"### {get_text('current_weather', global_lang)} {city}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(get_text("temperature", global_lang), f"{weather['temperature']}°C")
            with col2:
                st.metric(get_text("humidity", global_lang), f"{weather['humidity']}%")
            with col3:
                st.metric(get_text("rainfall", global_lang), f"{weather['rainfall']} mm")
            st.write(f"**{get_text('condition', global_lang)}:** {weather['description'].title()}")
        else:
            st.error(get_text("unable_weather", global_lang))
            st.info(get_text("check_connection", global_lang))

# ---------------------------
# Enhanced Emotion Support Chatbot with ChatGPT-like Theme
# ---------------------------
elif menu == get_text("menu_emotion", global_lang):
    # Enhanced CSS for WhatsApp-like chat theme
    st.markdown("""
    <style>
    /* Main chat container */
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .chat-header {
        text-align: center;
        color: white;
        margin-bottom: 20px;
        font-size: 24px;
        font-weight: 600;
    }

    /* Message container with WhatsApp-like background */
    .message-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 15px;
        background: #e5ddd5;
        background-image:
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
        border-radius: 15px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
    }

    /* Message bubbles */
    .user-bubble {
        background: #dcf8c6;
        background: linear-gradient(135deg, #dcf8c6 0%, #c3e88d 100%);
        padding: 8px 12px;
        border-radius: 8px 8px 4px 8px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-size: 14px;
        line-height: 1.3;
        position: relative;
    }

    .bot-bubble {
        background: white;
        padding: 8px 12px;
        border-radius: 8px 8px 8px 4px;
        margin: 5px 0;
        max-width: 70%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        border-left: 3px solid #667eea;
        font-size: 14px;
        line-height: 1.3;
        position: relative;
    }

    /* Message options */
    .message-options {
        opacity: 0;
        transition: opacity 0.3s ease;
        background: rgba(0,0,0,0.7);
        border-radius: 20px;
        padding: 5px 10px;
        position: absolute;
        top: -10px;
        right: 10px;
        z-index: 100;
    }

    .user-bubble:hover .message-options,
    .bot-bubble:hover .message-options {
        opacity: 1;
    }

    /* Emotion indicators */
    .emotion-indicator {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 500;
        margin-bottom: 3px;
    }

    .emotion-happy { background: #d4edda; color: #155724; }
    .emotion-sad { background: #f8d7da; color: #721c24; }
    .emotion-angry { background: #f5c6cb; color: #721c24; }
    .emotion-high-risk { background: #f8d7da; color: #721c24; animation: pulse 2s infinite; }

    /* Timestamps */
    .timestamp {
        font-size: 10px;
        color: #666;
        text-align: right;
        margin-top: 2px;
    }

    .timestamp-left {
        text-align: left;
    }

    /* Animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }

    /* Input area */
    .input-area {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 25px;
        margin-top: 15px;
        border: 1px solid #e0e0e0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .send-btn {
        background: #25d366;
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(37, 211, 102, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }

    .send-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(37, 211, 102, 0.6);
    }

    /* Feature cards */
    .feature-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Emergency alerts */
    .emergency-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 12px;
        text-align: center;
        animation: shake 0.5s ease-in-out;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }

    /* Action buttons */
    .action-btn {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 12px;
        margin: 0 5px;
    }

    .action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .delete-btn {
        background: #dc3545;
    }

    .delete-btn:hover {
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4);
    }

    /* Edit section */
    .edit-section {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Typing indicator */
    .typing-indicator {
        font-style: italic;
        color: #666;
        padding: 8px;
        text-align: center;
        font-size: 12px;
    }

    /* Scrollbar styling */
    .message-container::-webkit-scrollbar {
        width: 6px;
    }

    .message-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    .message-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }

    .message-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main container with ChatGPT-like design
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 AgriCare AI</div>', unsafe_allow_html=True)

    # Language selector
    lang_choice = st.selectbox("🌐 Language", languages, key="emotion_lang")

    # Initialize chat history and settings
    if "agri_history" not in st.session_state:
        st.session_state.agri_history = []
    if "emotion_messages" not in st.session_state:
        st.session_state.emotion_messages = []
    if "emergency_alerts_sent" not in st.session_state:
        st.session_state.emergency_alerts_sent = 0
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None
    if "edit_text" not in st.session_state:
        st.session_state.edit_text = ""
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    # Farmer context
    farmer_profile = st.session_state.get('farmer_profile', {})

    # Feature cards
    st.info("🧠 **AI-Powered**: Multi-tier AI: Cohere → Local LLM → DeepAI")
    st.info("🛡️ **Safety First**: Automatic emergency detection & WhatsApp alerts")
    st.info("🌍 **Multi-Language**: Support in 12+ Indian languages")

    # Chat messages container
    # Welcome message if no chat history
    if not st.session_state.agri_history and not st.session_state.emotion_messages:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; color: #666;">
            <h3>👋 Hey {farmer_name}, welcome to AgriCare AI!</h3>
            <p>I'm your friendly companion here to chat about farming, life, and whatever's on your mind. 🌱</p>
            <p style="font-size: 14px; color: #888;">Feel free to share anything - I'm here to listen and support you! 💚</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show conversation summary for returning users
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
        total_messages = len(st.session_state.agri_history) + len(st.session_state.emotion_messages)

        st.markdown(f"""
        <div style="text-align: center; padding: 15px; color: #666; background: #f8f9fa; border-radius: 10px; margin: 10px 0;">
            <p style="margin: 0; font-size: 14px;">👋 Welcome back, {farmer_name}! We've had {total_messages} messages in our conversation.</p>
            <p style="margin: 5px 0; font-size: 12px; color: #888;">I'm here whenever you need to continue our chat! 💬</p>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced ChatGPT-like conversation starters with more variety
    if not st.session_state.agri_history and not st.session_state.emotion_messages:
        st.markdown("### 💬 What would you like to talk about today?")

        # Create a grid of conversation starters
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🌾 Farming & Work**")
            farming_topics = [
                ("Crop Problems", "I'm having issues with my crops", "sad"),
                ("Weather Concerns", "The weather is worrying me", "sad"),
                ("Market Prices", "I want to talk about crop prices", "sad"),
                ("New Techniques", "I'm interested in new farming methods", "happy"),
                ("Harvest Success", "My harvest turned out great!", "happy")
            ]

            for topic_name, user_msg, emotion in farming_topics:
                if st.button(f"🌱 {topic_name}", key=f"farming_{topic_name.lower().replace(' ', '_')}", help=f"Talk about {topic_name.lower()}"):
                    bot_responses = {
                        "sad": [
                            f"I understand, {farmer_name}. Farming can be really challenging sometimes. Tell me more about what's been difficult for you.",
                            f"Oh, {farmer_name}, that sounds tough. I'm here to listen. What's been the biggest challenge lately?",
                            f"I hear you, {farmer_name}. Let's talk about this together. What specifically has been worrying you?"
                        ],
                        "happy": [
                            f"That's fantastic, {farmer_name}! 😊 I love hearing about farming successes. Tell me more about what went well!",
                            f"Wonderful news, {farmer_name}! 🌟 Your hard work is paying off. What made this harvest so successful?",
                            f"I'm so happy for you, {farmer_name}! 🎉 Success stories like yours inspire me. How did you achieve this?"
                        ]
                    }
                    bot_msg = bot_responses[emotion][0]
                    st.session_state.agri_history.append({
                        "user": user_msg,
                        "bot": bot_msg,
                        "emotion": emotion,
                        "timestamp": datetime.now()
                    })
                    # Clear input for conversation starters too
                    st.session_state.user_input_value = ""
                    st.session_state.clear_input = True
                    st.rerun()

        with col2:
            st.markdown("**💭 Personal & Emotional**")
            personal_topics = [
                ("Feeling Stressed", "I'm feeling really stressed lately", "sad"),
                ("Need Support", "I could use some emotional support", "sad"),
                ("Share Success", "I want to share some good news", "happy"),
                ("Family Matters", "I want to talk about family issues", "sad"),
                ("Just Chat", "I'd like to have a casual conversation", "happy")
            ]

            for topic_name, user_msg, emotion in personal_topics:
                if st.button(f"💙 {topic_name}", key=f"personal_{topic_name.lower().replace(' ', '_')}", help=f"Talk about {topic_name.lower()}"):
                    bot_responses = {
                        "sad": [
                            f"I'm here for you, {farmer_name}. 💙 It takes courage to reach out. What's been weighing on your mind?",
                            f"I can hear you're going through a difficult time, {farmer_name}. I'm right here with you. What's been the hardest part?",
                            f"Thank you for trusting me with this, {farmer_name}. 🌱 I'm listening. What would you like to talk about first?"
                        ],
                        "happy": [
                            f"That's wonderful, {farmer_name}! 😊 I love hearing from you. What's been bringing you joy lately?",
                            f"I'm so glad you're reaching out, {farmer_name}! 💚 What's new and exciting in your life?",
                            f"It's always a pleasure to chat with you, {farmer_name}! 🌟 What's been going well for you?"
                        ]
                    }
                    bot_msg = bot_responses[emotion][0]
                    st.session_state.agri_history.append({
                        "user": user_msg,
                        "bot": bot_msg,
                        "emotion": emotion,
                        "timestamp": datetime.now()
                    })
                    # Clear input for conversation starters too
                    st.session_state.user_input_value = ""
                    st.session_state.clear_input = True
                    st.rerun()

        # Quick emotion check-in
        st.markdown("---")
        st.markdown("### 😊 How are you feeling right now?")
        emotion_check = st.radio(
            "Quick check-in:",
            ["😊 Great!", "😐 Okay", "😔 Struggling", "😠 Frustrated", "Skip for now"],
            key="emotion_check",
            horizontal=True,
            label_visibility="collapsed"
        )

        if emotion_check and emotion_check != "Skip for now":
            emotion_map = {
                "😊 Great!": ("happy", "That's wonderful to hear! What made today good for you?"),
                "😐 Okay": ("sad", "I appreciate you sharing that. What's been on your mind lately?"),
                "😔 Struggling": ("sad", "I'm here for you. Would you like to talk about what's been difficult?"),
                "😠 Frustrated": ("angry", "I can sense your frustration. What happened that made you feel this way?")
            }

            detected_emotion, bot_msg = emotion_map[emotion_check]
            st.session_state.agri_history.append({
                "user": f"I'm feeling {emotion_check.lower()}",
                "bot": bot_msg,
                "emotion": detected_emotion,
                "timestamp": datetime.now()
            })
            # Clear input for emotion check-in too
            st.session_state.user_input_value = ""
            st.session_state.clear_input = True
            st.rerun()
    else:
        # Show conversation summary for returning users
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
        total_messages = len(st.session_state.agri_history) + len(st.session_state.emotion_messages)
        last_topic = "our conversation"  # Could be enhanced to detect topics

        st.markdown(f"""
        <div style="text-align: center; padding: 15px; color: #666; background: #f8f9fa; border-radius: 10px; margin: 10px 0;">
            <p style="margin: 0; font-size: 14px;">👋 Welcome back, {farmer_name}! We've had {total_messages} messages in our conversation.</p>
            <p style="margin: 5px 0; font-size: 12px; color: #888;">I'm here whenever you need to continue our chat! 💬</p>
        </div>
        """, unsafe_allow_html=True)

    # Combine both chat histories for display
    all_messages = st.session_state.agri_history + st.session_state.emotion_messages
    all_messages.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=False)

    # Display chat history using new chat bubble renderer
    for i, chat in enumerate(all_messages[-50:]):  # Show last 50 messages
        message_id = len(all_messages) - 50 + i if len(all_messages) > 50 else i

        # Format timestamp
        timestamp = chat.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        time_str = timestamp.strftime("%H:%M")

        # User message
        st.markdown(f"""
        <div style="background: #dcf8c6; padding: 8px 12px; border-radius: 8px 8px 4px 8px; margin: 5px 0; max-width: 70%; float: right; clear: both; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            {chat['user']}
            <div style="font-size: 10px; color: #666; text-align: right; margin-top: 2px;">✓✓ {time_str}</div>
        </div>
        <div style="clear: both;"></div>
        """, unsafe_allow_html=True)

        # Bot message
        emotion = chat.get('emotion', 'sad')
        emotion_emoji = {"happy": "😊", "sad": "😔", "angry": "😠", "high_risk": "🚨"}.get(emotion, "🤖")

        bot_message = f'<div style="margin-bottom: 5px;"><span style="background: #e9ecef; padding: 2px 6px; border-radius: 10px; font-size: 10px;">{emotion_emoji} {emotion.title()}</span></div>{chat["bot"]}'

        st.markdown(f"""
        <div style="background: white; padding: 8px 12px; border-radius: 8px 8px 8px 4px; margin: 5px 0; max-width: 70%; box-shadow: 0 1px 2px rgba(0,0,0,0.1); border-left: 3px solid #667eea;">
            {bot_message}
            <div style="font-size: 10px; color: #666; text-align: left; margin-top: 2px;">{time_str}</div>
        </div>
        """, unsafe_allow_html=True)

        # Emergency alert notification
        if chat.get('emergency_sent', False):
            st.error(f"🚨 EMERGENCY WHATSAPP ALERT SENT! WhatsApp sent to {chat.get('sms_count', 0)} family members • {time_str}")

        st.markdown("---")

    # Input area
    # Dynamic placeholder based on conversation history
    all_messages = st.session_state.agri_history + st.session_state.emotion_messages
    if not all_messages:
        placeholder_text = "Share what's on your mind today... 💭"
    else:
        farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
        placeholder_text = f"What's new with you, {farmer_name}? I'm here to listen... 💚"

    st.markdown("**💬 Your Message**")

    # Chat actions
    col1, col2, col3, col4 = st.columns([2, 2, 6, 2])

    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat", help="Clear all messages"):
            if st.session_state.agri_history or st.session_state.emotion_messages:
                st.session_state.agri_history = []
                st.session_state.emotion_messages = []
                st.session_state.emergency_alerts_sent = 0
                st.session_state.user_input_value = ""
                st.session_state.clear_input = False
                st.session_state.is_typing = False
                st.rerun()

    with col2:
        if st.button("💾 Export", key="export_chat", help="Export chat history"):
            all_messages = st.session_state.agri_history + st.session_state.emotion_messages
            if all_messages:
                chat_text = "AgriCare AI Chat History\n\n"
                for chat in sorted(all_messages, key=lambda x: x.get('timestamp', datetime.now())):
                    timestamp = chat.get('timestamp', datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    chat_text += f"You ({time_str}):\n{chat['user']}\n\n"
                    chat_text += f"AgriCare AI ({time_str}):\n{chat['bot']}\n\n"
                    if chat.get('emergency_sent'):
                        chat_text += f"🚨 EMERGENCY WHATSAPP ALERT SENT to {chat.get('sms_count', 0)} family members\n\n"
                    chat_text += "---\n\n"

                st.download_button(
                    label="📥 Download Chat",
                    data=chat_text,
                    file_name="agricare_ai_chat_history.txt",
                    mime="text/plain",
                    key="download_chat"
                )

    with col3:
        # Dynamic placeholder based on conversation history
        all_messages = st.session_state.agri_history + st.session_state.emotion_messages
        if not all_messages:
            placeholder_text = "Share what's on your mind today... 💭"
        else:
            farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'
            placeholder_text = f"What's new with you, {farmer_name}? I'm here to listen... 💚"

        user_input = st.text_input(
            "Type your message",
            placeholder=placeholder_text,
            key="emotion_input",
            label_visibility="collapsed",
            value=st.session_state.get('user_input_value', '')
        )

        # Clear input value after processing
        if st.session_state.get('clear_input', False):
            st.session_state.user_input_value = ""
            st.session_state.clear_input = False

    with col4:
        if st.button("📤 Send", key="send_emotion", help="Send message", use_container_width=True):
            if user_input.strip():
                # Detect emotion first
                detected_emotion = detect_emotion(user_input)

                # Show typing indicator
                st.session_state.is_typing = True

                # Get AI response with conversation context (use combined history)
                all_messages = st.session_state.agri_history + st.session_state.emotion_messages

                # Try multiple AI services in order of preference
                bot_response = None

                # 1. Try Cohere API if available
                if COHERE_API_KEY:
                    try:
                        bot_response = get_cohere_response(
                            user_input,
                            detected_emotion,
                            lang_choice,
                            farmer_profile,
                            all_messages
                        )
                        print("Using Cohere API response")
                    except Exception as e:
                        print(f"Cohere API Error: {e}")
                        bot_response = None

                # 2. Try local free LLM
                if bot_response is None and free_chat_model is not None:
                    try:
                        bot_response = get_free_llm_response(
                            user_input,
                            detected_emotion,
                            lang_choice,
                            farmer_profile,
                            all_messages
                        )
                        print("Using local free LLM response")
                    except Exception as e:
                        print(f"Local LLM Error: {e}")
                        bot_response = None

                # 3. Final fallback
                if bot_response is None:
                    bot_response = get_chatgpt_style_fallback(detected_emotion, lang_choice, farmer_profile, user_input, all_messages)
                    print("Using fallback response")

                # Hide typing indicator
                st.session_state.is_typing = False

                # Check for emergency situation
                emergency_sent = False
                sms_count = 0

                if detected_emotion == "high_risk" and farmer_profile:
                    sms_count = send_emergency_whatsapp(farmer_profile, "Current Location", lang_choice)
                    if sms_count > 0:
                        emergency_sent = True
                        st.session_state.emergency_alerts_sent += 1

                # Prevent duplicate message appending
                message_exists = False
                for chat in all_messages:
                    if (chat.get('user') == user_input and
                        chat.get('timestamp') and
                        (datetime.now() - chat['timestamp']).total_seconds() < 5):  # Within 5 seconds
                        message_exists = True
                        break

                if not message_exists:
                    # Add to chat history (use emotion_messages for new messages)
                    st.session_state.emotion_messages.append({
                        "user": user_input,
                        "bot": bot_response,
                        "emotion": detected_emotion,
                        "timestamp": datetime.now(),
                        "emergency_sent": emergency_sent,
                        "sms_count": sms_count
                    })

                # Clear the input after sending
                st.session_state.user_input_value = ""
                st.session_state.clear_input = True

                # Rerun to update chat
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # End input container

    # Enhanced ChatGPT-like Conversation Insights using new renderers
    all_messages = st.session_state.agri_history + st.session_state.emotion_messages
    if all_messages:
        with st.expander("📊 Conversation Insights", expanded=False):
            total_messages = len(all_messages)
            emotions_detected = [chat.get('emotion', 'neutral') for chat in all_messages if chat.get('emotion')]
            emotion_counts = {}
            for emotion in emotions_detected:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"

            farmer_name = farmer_profile.get('name', 'friend') if farmer_profile else 'friend'

            # Calculate conversation patterns
            conversation_length = len(all_messages)
            emergency_alerts = st.session_state.emergency_alerts_sent

            # Use stats grid for conversation metrics
            stats = {
                "Total Messages": (total_messages, "💬"),
                "Primary Emotion": (most_common_emotion.title(), "😊" if most_common_emotion == "happy" else "😔" if most_common_emotion == "sad" else "😠" if most_common_emotion == "angry" else "🚨"),
                "Emergency Alerts": (emergency_alerts, "📱")
            }
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", stats["Total Messages"][0], delta=None)
            col2.metric("Primary Emotion", stats["Primary Emotion"][0], delta=None)
            col3.metric("Emergency Alerts", stats["Emergency Alerts"][0], delta=None)

            # Generate personalized insights
            insights = []

            if most_common_emotion == "happy":
                insights.append("🌟 You're showing a positive outlook - that's wonderful!")
            elif most_common_emotion == "sad":
                insights.append("💙 I notice you've been going through some challenges lately")
            elif most_common_emotion == "angry":
                insights.append("😠 It seems like frustration has been a common theme")
            elif most_common_emotion == "high_risk":
                insights.append("🚨 I've detected some concerning moments in our conversation")

            if emergency_alerts > 0:
                insights.append(f"📱 Emergency WhatsApp support was activated {emergency_alerts} time(s) - help is available")

            if conversation_length > 10:
                insights.append("🎯 We've had a meaningful conversation - I'm here whenever you need to continue")

            # Time-based insights
            if conversation_length > 1:
                sorted_messages = sorted(all_messages, key=lambda x: x.get('timestamp', datetime.now()))
                first_message = sorted_messages[0]['timestamp']
                last_message = sorted_messages[-1]['timestamp']
                if isinstance(first_message, str):
                    first_message = datetime.fromisoformat(first_message)
                if isinstance(last_message, str):
                    last_message = datetime.fromisoformat(last_message)

                conversation_duration = (last_message - first_message).total_seconds()
                hours_active = conversation_duration / 3600

                if hours_active > 24:
                    insights.append("⏰ Our conversation has spanned multiple days - consistency shows strength")

            if insights:
                insights_content = "<br>".join(f"• {insight}" for insight in insights)
                st.info(f"<strong>💡 Insights:</strong><br>{insights_content}")

            # Remember message
            remember_content = f"""
            <strong>💚 Remember:</strong> {farmer_name}, I'm always here for you. Whether you want to talk about farming challenges,
            share good news, or just need someone to listen - I'm just a message away. Your well-being matters to me! 🌱
            """
            st.success(remember_content)

    # Enhanced typing indicator using new renderer
    if st.session_state.get('is_typing', False):
        with st.spinner("?? AI is typing..."): st.empty()
        # Force a small delay to show typing indicator
        import time
        time.sleep(0.5)

    # Emergency stats
    if st.session_state.emergency_alerts_sent > 0:
        st.warning(f"Emergency WhatsApp alerts sent: {st.session_state.emergency_alerts_sent}")

    # Helpline information
    with st.expander("🆘 Emergency Helplines"):
        helpline_content = """
        <strong>India Emergency Helplines:</strong><br><br>
        • <strong>Mental Health:</strong> 1800-121-4559 (AASRA)<br>
        • <strong>Farmer Helpline:</strong> 1800-120-0024 (Kisan Call Centre)<br>
        • <strong>Suicide Prevention:</strong> 9152987821 (Vandrevala Foundation)<br>
        • <strong>Police:</strong> 100 | <strong>Ambulance:</strong> 108<br><br>
        <strong>Remember:</strong> You're not alone. Help is always available! 🌟
        """
        st.markdown(helpline_content)

# ---------------------------
# Emergency Alert
# ---------------------------
elif menu == get_text("menu_emergency", global_lang):
    st.subheader("🚨 " + get_text("emergency_alert", global_lang))
    if "farmer_profile" in st.session_state:
        profile = st.session_state.farmer_profile
        st.write(f"**{get_text('farmer', global_lang)}:** {profile['name']} (Age: {profile['age']})")
        st.write(f"**{get_text('emergency_contacts', global_lang)}:**")
        st.write(f"- {profile['family1']['name']}: {profile['family1']['phone']}")
        st.write(f"- {profile['family2']['name']}: {profile['family2']['phone']}")

    location = st.text_input(get_text("location", global_lang))
    lang_choice = st.selectbox(get_text("select_language", global_lang), languages, key="alert_lang")

    if st.button(get_text("send_alert", global_lang)):
        if "farmer_profile" in st.session_state:
            name = st.session_state.farmer_profile["name"]
        else:
            name = "Farmer"
        base_msg = f"⚠ Emergency Alert for 👨‍🌾 {name} at 📍 {location}!"
        alert_msg = emotion_translations.get(lang_choice, lambda x:x)(base_msg)
        st.warning(alert_msg)
        st.balloons()
