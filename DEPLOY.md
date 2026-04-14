# AgriDream Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

1. **Push code to GitHub** (already done ✓)

2. **Go to:** https://share.streamlit.io

3. **Login with GitHub** and select your repository:
   - Repository: `thirisha2006-S/gitAgridream25`
   - Branch: `main`
   - Main file: `app.py`

4. **Advanced Settings:**
   - Python version: 3.11
   - Requirements file: `requirements.txt`

5. **Click Deploy!** ✓

---

### Option 2: Hugging Face Spaces (FREE)

1. **Go to:** https://huggingface.co/spaces

2. **Create New Space:**
   - Name: `agridream`
   - Type: Streamlit
   - Visibility: Public

3. **Connect to GitHub** and select your repo

4. **Deploy!** ✓

---

### Option 3: Render (FREE)

1. **Go to:** https://render.com

2. **Create Web Service:**
   - Connect GitHub repo
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT`

3. **Deploy!** ✓

---

## 📋 Pre-Deployment Checklist

- [x] requirements.txt - Updated
- [x] app.py - Main file ready
- [x] database.py - SQLite module
- [x] .env - API keys configured
- [ ] **Add API keys to deployment secrets**

---

## 🔑 Required API Keys (for full functionality)

Add these in your deployment platform's "Secrets" or "Environment Variables":

| Key | Purpose | Get from |
|-----|---------|----------|
| OPENAI_API_KEY | AI Chat | platform.openai.com |
| DEEP_AI_API_KEY | Image AI | deepai.org |
| OPENWEATHER_API_KEY | Weather | openweathermap.org/api |
| PLANTNET_API_KEY | Disease ID | my.plantnet.org |
| DATA_GOV_IN_API_KEY | Price Data | data.gov.in |

---

## 🌐 After Deployment

Your app will be live at:
- **Streamlit:** `https://[your-name].streamlit.app`
- **HuggingFace:** `https://huggingface.co/spaces/[your-name]/agridream`
- **Render:** `https://[your-service].onrender.com`

---

## 📱 For Wide Use

1. **Share the URL** with farmers
2. **No installation needed** - works in browser
3. **Works on mobile** too!

---

## 🔧 Troubleshooting

**Issue:** App crashes on startup
→ Check Python version (use 3.11)
→ Check requirements.txt

**Issue:** Weather not working
→ Add OPENWEATHER_API_KEY in secrets

**Issue:** Database error
→ SQLite works automatically - no setup needed!