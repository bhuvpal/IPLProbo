import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))

st.set_page_config("IPL Match Predictor",layout='centered',page_icon="🏏")
st.title("🏏IPL matches Prediction")

batballcol = ['Royal Challengers Bangalore', 'Delhi Daredevils',
       'Mumbai Indians', 'Kings XI Punjab', 'Kolkata Knight Riders',
       'Sunrisers Hyderabad', 'Rajasthan Royals', 'Chennai Super Kings']


city = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']


st.divider()


col1,col2 = st.columns(2)
with col1:
     batting_team = st.selectbox("Please Select Batting team",sorted(batballcol)+["None"],index=len(batballcol))
with col2:
     bowling_team = st.selectbox("Please Select Bowling team",sorted(batballcol)+["None"],index=len(batballcol))


city = st.selectbox("Please Select Place city",sorted(city)+["None"],index=len(city))

target = st.number_input("Please Select the target")


col3,col4,col5 = st.columns(3)
with col3:
     runs = st.number_input("Please Select score")
with col4:
     overs = st.number_input("How many Over completed")
with col5:
     wickets = st.number_input("Please Select Wickets")

st.divider()

if st.button("Predict"):
     runs_left = target - runs
     balls_left = 120 - overs*6
     wickets_left = 10 - wickets
     crr = runs/overs
     rrr = runs_left*6/balls_left


     input_data = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                                "city":[city],"target":[target],"runs_left":[runs_left],
                                "balls_left":[balls_left],"wickets_left":[wickets_left],
                                "crr":[crr],"rrr":[rrr]})
     

    
     Result = pipe.predict_proba(input_data)
     lossing = Result[0][0]
     wining = Result[0][1]
     proba = [100 - lossing*100,100 - wining*100]
     lablee = [batting_team,bowling_team]


     fig = plt.figure(figsize=(10, 6))
     plt.pie(proba,labels=lablee,autopct="%1.1f%%",shadow=True)
     st.pyplot(fig)


     st.header("Probability of winnig")
     st.header(batting_team + " - " + str(round(wining*100)) + "%")
     st.header(bowling_team + " - " + str(round(lossing*100)) + "%")
     