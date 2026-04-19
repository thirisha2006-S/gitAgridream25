# Disease Detection using Plant.id API
# ---------------------------
elif menu == get_text("menu_disease", global_lang):
    st.subheader("🌿 " + get_text("menu_disease", global_lang))
    st.write("📷 Upload a photo of your plant leaf to detect diseases")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Plant Image", use_container_width=True)
        
        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analyzing plant image..."):
                try:
                    image_bytes = uploaded_file.getvalue()
                    
                    import base64
                    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    PLANT_ID_API_KEY = os.getenv('PLANT_ID_API_KEY', 'LiobPhyMKoo9i5L0oR8otFzxRgo5FuDzcMWFIaXn1JBwVyMAFv')
                    
                    headers = {'Content-Type': 'application/json', 'Api-Key': PLANT_ID_API_KEY}
                    data = {
                        'images': [f'data:image/jpeg;base64,{img_base64}'],
                        'latitude': 34.05, 'longitude': -118.25,
                        'datetime': int(datetime.now().timestamp())
                    }
                    
                    response = requests.post('https://api.plant.id/v2/identify', headers=headers, json=data, timeout=30)
                    
                    if response.status_code == 200:
                        result_data = response.json()
                        suggestions = result_data.get('suggestions', [])
                        
                        if suggestions:
                            top_match = suggestions[0]
                            plant_name = top_match.get('plant_name', 'Unknown Plant')
                            disease_name = top_match.get('disease_name', 'Healthy')
                            probability = int(top_match.get('probability', 0) * 100)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Detection Results")
                            st.markdown(f"**🌱 Plant:** {plant_name}")
                            st.markdown(f"**🦠 Status:** {disease_name}")
                            st.markdown(f"**Confidence:** {probability}%")
                            
                            if 'healthy' in disease_name.lower() or probability > 70:
                                st.success("✅ Plant appears healthy!")
                            else:
                                st.warning("⚠️ Disease detected")
                            
                            st.markdown("### 💊 Recommended Treatment")
                            st.info("Remove infected leaves, apply appropriate fungicide, improve air circulation.")
                            st.caption("Powered by Plant.id API")
                        else:
                            st.warning("No results found. Try a clearer image.")
                    else:
                        st.warning("API error. Please try again.")
                        
                except Exception as e:
                    st.warning(f"Error: {str(e)[:100]}")
                    st.info("Please try again with a clearer image.")

# ============================================================
# AgriCare AI - Smart Farming Assistant  
# ============================================================
