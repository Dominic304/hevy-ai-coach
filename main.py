import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI

# --- 1.Page config ---
st.set_page_config(page_title="Hevy AI Coach", layout="wide")
st.title("Hevy AI Workout Analyzer & Visualizer")
st.write("Provide your API keys and fetch your data to get personalized programming advice.")

# --- 2.User interface ---
with st.sidebar:
    st.header("Configuration")
    hevy_api_key = st.text_input("Hevy API Key", type="password", help="Get this from hevy.com/settings?developer")
    openrouter_api_key = st.text_input("OpenRouter / AI API Key", type="password")
    months_to_analyze = st.slider("Months of Data to Analyze", 1, 24, 3)
    user_goal = st.text_area("What is your current main goal? (e.g., Hypertrophy, fix bench plateau)",value="Hypertrophy", height=100)



    to_graph = st.checkbox("Graph my training metrics", value=True)

    st.header("Customize Coach Prompt")
    # This text area loads the default base prompt, but allows you to completely edit it live
    custom_system_prompt = st.text_area(
        "Edit Coach Rules:",
        value=(
            "You are an elite strength and conditioning coach.\n"
            "Analyze the following workout history. Identify volume imbalances, "
            "track progressive overload consistency, and provide exactly 3 specific, "
            "actionable programming modifications tailored to the user's goals."
        ),
        height=150
    )

# --- 3.Data,parsing ---
def fetch_and_clean_hevy_data(api_key, months):
    """Fetches paginated workouts from Hevy and strips out unnecessary metadata."""
    
    headers = {"api-key": api_key.strip()}
    base_url = "https://api.hevyapp.com/v1/workouts"
    
    cutoff_date = datetime.now() - timedelta(days=30 * months)
    
    all_workouts = []
    page = 1
    
    while True:
        params = {
            "page": page,
            "pageSize": 10 
        }
        
        response = requests.get(base_url, headers=headers, params=params)
        
        # If 404 on a page AFTER page 1 >= finished fetching all available data
        if response.status_code == 404 and page > 1:
            break
            
        if response.status_code != 200:
            st.error(f"Hevy API Error: {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        workouts = data.get("workouts", [])
        
        page_count = data.get("page_count", 1)
        
        if not workouts:
            break
            
        reached_cutoff = False
        for workout in workouts:
            workout_date_str = workout.get("start_time", workout.get("created_at"))
            if not workout_date_str:
                continue
                
            workout_date = datetime.fromisoformat(workout_date_str.replace('Z', '+00:00'))
            
            if workout_date.replace(tzinfo=None) < cutoff_date:
                reached_cutoff = True
                break
                
            # Clean data
            cleaned_workout = {
                "date": workout_date.strftime("%Y-%m-%d"),
                "name": workout.get("name", "Workout"),
                "exercises": []
            }
            
            for exercise in workout.get("exercises", []):
                ex_data = {
                    "title": exercise.get("title", "Unknown Exercise"),
                    "sets": []
                }
                for s in exercise.get("sets", []):
                    ex_data["sets"].append({
                        "weight": s.get("weight_kg", 0),
                        "reps": s.get("reps", 0),
                    })
                cleaned_workout["exercises"].append(ex_data)
                
            all_workouts.append(cleaned_workout)
            
        # Stop fetching if we reached the date limit OR if we are on your last available page
        if reached_cutoff or page >= page_count:
            break
            
        page += 1
    return all_workouts
    

# --- 4.Feeding data to AI---
if st.button("Analyze My Training", type="primary"):
    if not hevy_api_key or not openrouter_api_key:
        st.warning("Please provide both API keys in the sidebar.")
    elif not user_goal:
        st.warning("Please provide a training goal so the AI knows what to optimize for.")
    else:
        with st.spinner("Fetching data from Hevy..."):
            workout_data = fetch_and_clean_hevy_data(hevy_api_key, months_to_analyze)
        st.success(f"Successfully loaded {len(workout_data)} workouts from the last {months_to_analyze} months.")
        if workout_data:
            
            
            # --- 5. Data visualization engine ---
            if to_graph:
             
             st.markdown("##Your Training Metrics")
             
             # Flatten the nested workout details into a list for pandas calculation
             chart_records = []
             for w in workout_data:
                 total_workout_volume = 0
                 for ex in w["exercises"]:
                     for s in ex["sets"]:
                         # Safely fallback to 0 if Hevy returns null for bodyweight exercises
                         weight = s.get("weight") or 0
                         reps = s.get("reps") or 0
                         total_workout_volume += (weight * reps)
                 
                 chart_records.append({
                     "Date": pd.to_datetime(w["date"]),
                     "Volume (kg)": total_workout_volume,
                     "Workouts": 1
                 })
             
             df = pd.DataFrame(chart_records)
             
             if not df.empty:
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     st.subheader("Total Session Volume Over Time")
                     df_sorted = df.sort_values("Date")
                     st.line_chart(data=df_sorted, x="Date", y="Volume (kg)")
                     
                 with col2:
                     st.subheader("Workout Frequency (Weekly Count)")
                     df_weekly = df.resample('W', on='Date').sum().reset_index()
                     st.bar_chart(data=df_weekly, x="Date", y="Workouts")
                
                
            # --- 6. AI Analysis Engine ---
            with st.spinner("Analyzing your programming..."):
                # Configure OpenAI client to point to OpenRouter
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                )
                
                # Construct the combined prompt using your editable base prompt
                prompt = f"""
                User's Primary Training Goal: {user_goal}
                
                Coach Analysis Guidelines:
                {custom_system_prompt}
                
                Below is their parsed workout history for the last {months_to_analyze} months.
                
                Data:
                {workout_data}
                """
                
                # Call the OpenRouter Free Model Router
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                st.markdown("---")
                st.markdown("### Coach's Analysis")
                st.write(response.choices[0].message.content)