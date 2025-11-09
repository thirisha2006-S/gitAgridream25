from pyngrok import ngrok
import subprocess
import time

# Start Streamlit app
print("Starting Streamlit app...")
process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.address', '0.0.0.0'])

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
print("Creating ngrok tunnel...")
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")

# Keep the tunnel alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping tunnel...")
    ngrok.disconnect(public_url)
    ngrok.kill()
    process.terminate()