"""
Chat UI Module - WhatsApp-style chat interface
"""
import streamlit as st
from datetime import datetime
import pytz

def render_chat_header(farmer_name):
    """Render chat header with farmer name and time"""
    ist = pytz.timezone('Asia/Kolkata')
    ist_time = datetime.now(ist).strftime("%d-%m-%Y %H:%M")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #075E54, #128C7E); 
                padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h2 style="color: white; margin: 0;">🌾 AgriCare AI</h2>
        <p style="color: #D1D1D1; margin: 5px 0 0 0; font-size: 12px;">
            🕐 India: {ist_time} | 👤 {farmer_name}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_bubble(message, is_user=True, emotion="happy"):
    """Render a single chat bubble (WhatsApp style)"""
    
    # Emoji based on emotion
    emotion_emoji = {
        "happy": "😊",
        "sad": "😔", 
        "angry": "😠",
        "high_risk": "🚨"
    }
    emoji = emotion_emoji.get(emotion, "💚")
    
    if is_user:
        # User message - right aligned, green background
        bubble_html = f"""
        <div style="display: flex; justify-content: flex-end; margin: 8px 0;">
            <div style="background: #DCF8C6; 
                        padding: 10px 15px; 
                        border-radius: 10px 10px 0 10px;
                        max-width: 70%;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                <p style="margin: 0; color: #000; font-size: 14px;">{message}</p>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 10px; text-align: right;">✓</p>
            </div>
        </div>
        """
    else:
        # AI message - left aligned, white background
        bubble_html = f"""
        <div style="display: flex; justify-content: flex-start; margin: 8px 0;">
            <div style="background: #FFFFFF; 
                        padding: 10px 15px; 
                        border-radius: 10px 10px 10px 0;
                        max-width: 70%;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                <p style="margin: 0; color: #000; font-size: 14px;">{emoji} {message}</p>
            </div>
        </div>
        """
    
    st.markdown(bubble_html, unsafe_allow_html=True)


def render_welcome_message(farmer_name):
    """Render welcome message"""
    welcome_html = f"""
    <div style="text-align: center; padding: 30px; background: #f0f0f0; border-radius: 15px; margin: 20px 0;">
        <h3 style="color: #075E54;">👋 Namaste, {farmer_name}!</h3>
        <p style="color: #666;">I'm AgriCare AI, your farming companion.</p>
        <p style="color: #888; font-size: 13px;">Type a message below to start chatting...</p>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)


def render_emotion_indicator(current_emotion):
    """Render current emotion status panel"""
    emotion_data = {
        "happy": ("😊", "Feeling Good", "green"),
        "sad": ("😔", "Feeling Low", "orange"),
        "angry": ("😠", "Frustrated", "red"),
        "high_risk": ("🚨", "Need Support", "red")
    }
    
    emo_emoji, emo_status, emo_color = emotion_data.get(current_emotion, ("😊", "Good", "green"))
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin: 10px 0;">
        <strong>🧠 Your Status:</strong> <span style="color: {emo_color};">{emo_emoji} {emo_status}</span>
    </div>
    """, unsafe_allow_html=True)


def render_help_lines():
    """Render emergency help lines"""
    st.markdown("""
    <div style="background: #FFF3CD; padding: 10px; border-radius: 8px; margin: 10px 0;">
        <strong>📞 Help Lines:</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
        <li>🧠 iCall: 9152987821</li>
        <li>🚔 Police: 100</li>
        <li>🌾 Kisan: 1800-180-1551</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons"""
    st.markdown("""
    <div style="margin: 15px 0;">
        <p><strong>⚡ Quick Actions:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("🌱 Crop Help", use_container_width=True)
    with col2:
        st.button("🌤️ Weather", use_container_width=True)
    with col3:
        st.button("💰 Prices", use_container_width=True)