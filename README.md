# AgriDream Smart Farming Assistant 🌾

A comprehensive AI-powered farming assistant built with Streamlit, featuring crop recommendation, price forecasting, weather information, and emotional support for farmers.

## Features

- 🌾 **Crop Recommendation**: AI-powered crop suggestions based on soil and climate conditions
- 💹 **Price Forecasting**: Live crop price information and market insights
- 🌤️ **Weather Information**: Real-time weather data for farming decisions
- 🤖 **AgriCare AI**: Multi-language emotional support chatbot with emergency alert system
- 🚨 **Emergency Alert System**: WhatsApp alerts to family members in crisis situations
- 🌍 **Multi-language Support**: Available in 12+ Indian languages

## Deployment on Streamlit Community Cloud

### Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **API Keys**: Obtain the following API keys:
   - OpenAI API Key (for ChatGPT-like responses)
   - Cohere API Key (primary AI service)
   - DeepAI API Key (fallback AI service)
   - OpenWeather API Key (for weather data)
   - CallMeBot API Key (for WhatsApp emergency alerts)

### Step-by-Step Deployment

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account

3. **Deploy your app**:
   - Click "New app"
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"

4. **Configure Secrets**:
   - In your Streamlit Cloud dashboard, go to your app settings
   - Navigate to "Secrets" section
   - Add the following secrets (copy from `secrets.toml`):

   ```
   OPENAI_API_KEY = "your_openai_api_key_here"
   DEEP_AI_API_KEY = "your_deepai_api_key_here"
   COHERE_API_KEY = "your_cohere_api_key_here"
   OPENWEATHER_API_KEY = "your_openweather_api_key_here"
   CALLMEBOT_API_KEY = "your_callmebot_api_key_here"
   CALLMEBOT_PHONE = "your_callmebot_phone_number_here"
   ```

5. **Redeploy**: After adding secrets, redeploy your app for changes to take effect

### API Keys Setup

#### OpenAI API Key
- Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- Create a new API key
- Add to Streamlit secrets as `OPENAI_API_KEY`

#### Cohere API Key
- Visit [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
- Create a new API key
- Add to Streamlit secrets as `COHERE_API_KEY`

#### DeepAI API Key
- Visit [DeepAI](https://deepai.org/)
- Sign up and get your API key
- Add to Streamlit secrets as `DEEP_AI_API_KEY`

#### OpenWeather API Key
- Visit [OpenWeatherMap](https://openweathermap.org/api)
- Sign up for free tier
- Get your API key
- Add to Streamlit secrets as `OPENWEATHER_API_KEY`

#### CallMeBot WhatsApp API
- Visit [CallMeBot](https://www.callmebot.com/)
- Follow instructions to set up WhatsApp API
- Get your API key and phone number
- Add to Streamlit secrets as `CALLMEBOT_API_KEY` and `CALLMEBOT_PHONE`

### File Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System packages (for cloud deployment)
├── secrets.toml             # Secrets configuration template
├── agmarknet_prices.csv     # Crop price data
├── crop_recommendation.csv  # Crop recommendation dataset
├── README.md                # This file
└── .gitattributes          # Git attributes
```

### Important Notes

- **Free Tier**: Streamlit Community Cloud is completely free for public apps
- **Resource Limits**: Free tier has some limitations on compute resources
- **Secrets**: All API keys must be configured in Streamlit Cloud secrets, not in code
- **Data Files**: CSV files are automatically included in deployment
- **Dependencies**: All packages in `requirements.txt` will be installed automatically

### Troubleshooting

1. **App not loading**: Check that all required secrets are configured
2. **API errors**: Verify your API keys are valid and have sufficient credits
3. **Data not loading**: Ensure CSV files are in the repository root
4. **Dependencies issues**: Check that all packages in `requirements.txt` are compatible with Streamlit Cloud

### Support

For issues with deployment or the application, please check:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)

---

**Built with ❤️ for Indian farmers** 🌾