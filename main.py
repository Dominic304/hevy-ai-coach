import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI

# ---------------------Page config-----------------------
st.set_page_config(page_title="Hevy AI Coach", layout="wide")
st.title("Hevy AI Workout Analyzer & Visualizer")
st.write("Provide your API keys and fetch your data to get personalized programming advice.")

# ----------------------User interface------------------------
with st.sidebar:
    st.header("Configuration")
    hevy_api_key = st.text_input("Hevy API Key", type="password", help="Get this from hevy.com/settings?developer")
    openrouter_api_key = st.text_input("OpenRouter / AI API Key", type="password")
    months_to_analyze = st.slider("Months of Data to Analyze", 1, 24, 3)
    user_goal = st.text_area("What is your current main goal? (e.g., Hypertrophy, fix bench plateau)", value="Hypertrophy", height=100)
    
    to_graph = st.checkbox("Graph my training metrics", value=True)

    st.header("Customize Coach Prompt")
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

# ----------------------Data parsing---------------------------
def fetch_and_clean_hevy_data(api_key, months):
    """Fetches paginated workouts from Hevy and strips out unnecessary metadata."""
    headers = {"api-key": api_key.strip()}
    base_url = "https://api.hevyapp.com/v1/workouts"
    cutoff_date = datetime.now() - timedelta(days=30 * months)
    
    all_workouts = []
    page = 1
    
    while True:
        params = {"page": page, "pageSize": 10}
        response = requests.get(base_url, headers=headers, params=params)
        
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
                        "weight": s.get("weight_kg") or 0,
                        "reps": s.get("reps") or 0,
                    })
                cleaned_workout["exercises"].append(ex_data)
                
            all_workouts.append(cleaned_workout)
            
        if reached_cutoff or page >= page_count:
            break
            
        page += 1
    return all_workouts

# ---------------------------------- STREAMLIT MEMORY  ------------------------------------
if "workout_data" not in st.session_state:
    st.session_state.workout_data = None
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None


# ---------------------------------- Execution Flow ---------------------------------------
if st.button("Analyze My Training", type="primary"):
    if not hevy_api_key or not openrouter_api_key:
        st.warning("Please provide both API keys in the sidebar.")
    elif not user_goal:
        st.warning("Please provide a training goal so the AI knows what to optimize for.")
    else:
        with st.spinner("Fetching data from Hevy..."):
            st.session_state.workout_data = fetch_and_clean_hevy_data(hevy_api_key, months_to_analyze)
        
        if st.session_state.workout_data:
            with st.spinner("AI is analyzing your metrics..."):
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                )
                
                full_prompt = f"""
                User's Primary Training Goal: {user_goal}
                Coach Analysis Guidelines:
                {custom_system_prompt}
                Data:
                {st.session_state.workout_data}
                """
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": full_prompt}]
                )
                st.session_state.ai_analysis = response.choices[0].message.content


# --------------------------------------- RENDER UI ------------------------------------
if st.session_state.workout_data:
    st.success(f"Successfully loaded {len(st.session_state.workout_data)} workouts from the last {months_to_analyze} months.")
    
    st.markdown("Coach's Analysis")
    st.write(st.session_state.ai_analysis)
    
    if to_graph:
        st.markdown("---")
        st.markdown("Exercise Progress Charts")
        st.write("Track your progressive overload over time.")
        
        exercise_records = []
        for w in st.session_state.workout_data:
            date = pd.to_datetime(w["date"])
            for ex in w["exercises"]:
                title = ex["title"]
                weights = [s.get("weight") or 0 for s in ex["sets"]]
                max_weight_for_session = max(weights) if weights else 0
                
                if max_weight_for_session > 0:
                    exercise_records.append({
                        "Date": date,
                        "Exercise": title,
                        "Max Weight (kg)": max_weight_for_session
                    })
        
        df_ex = pd.DataFrame(exercise_records)
        
        if not df_ex.empty:
            unique_exercises = sorted(df_ex["Exercise"].unique())
            selected_ex = st.selectbox("Select an exercise to view your strength progress:", unique_exercises)
            
            # Filter for the selected exercise
            df_filtered = df_ex[df_ex["Exercise"] == selected_ex]
            
            #Group by date 
            df_filtered = df_filtered.groupby("Date")["Max Weight (kg)"].max().reset_index()
            df_filtered = df_filtered.sort_values("Date")
            
            #Handle single data point vs multiple data points
            if len(df_filtered) == 1:
                st.info(f"You have only logged '{selected_ex}' on one day during this timeframe. Keep training it to see a trend line!")
                st.scatter_chart(data=df_filtered, x="Date", y="Max Weight (kg)")
            else:
                st.line_chart(df_filtered.set_index("Date")["Max Weight (kg)"])
        else:
            st.info("No weighted exercise data found to chart.")