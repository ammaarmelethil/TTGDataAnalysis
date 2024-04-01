# To see app, run 'streamlit run TTG_Insights_Dash.py' in terminal
# Make sure all packages below are installed

import streamlit as st
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import openai

# Load DataFrame from the pickle file
with open('my_dataframe.pkl', 'rb') as file:
    df = pickle.load(file)

## STREAMLIT APP ##

## FILTERS AND CONFUGRATION #############################################################################################################################################################################################

# AI ASSISTANT - Comment out this code before 'Dashboard' to view GUI without having an OpenAI API key
openai.api_key = ""

# Add a section for OpenAI-powered insights
st.title('Trade Insights Assistant 🤖')

# Add a text input for the user to submit their query
user_query = st.text_input('Enter your question here:', '')

# Function to call the OpenAI API
def ask_openai(question, df):
    try:
        # Combine the user's question with the data summary
        prompt = f"Question: {question}\n\nData Summary:\n{df}\n\nAnswer:"

        # Make the API call
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150,
            n=1,

        )
        # Extract the text from the response object
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return str(e)


# Display the answer to the user's query
if user_query:
    with st.spinner('Getting insights from Data...'):
        answer = ask_openai(user_query, df)
        st.write(answer)



# DASHBOARD ############################################################################################################################################################################################################

st.title('Trade Insights Dashboard 📈')

# Sidebar filters for controllable variables
st.sidebar.header('Filter Options')

# Adding date-time range filters
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime(df['When'].dt.date.min()),
                                   min_value=pd.to_datetime(df['When'].dt.date.min()),
                                   max_value=pd.to_datetime(df['When'].dt.date.max()))
start_time = st.sidebar.time_input('Start time', value=pd.to_datetime('00:00').time())
end_date = st.sidebar.date_input('End date', value=pd.to_datetime(df['When'].dt.date.max()),
                                 min_value=pd.to_datetime(df['When'].dt.date.min()),
                                 max_value=pd.to_datetime(df['When'].dt.date.max()))
end_time = st.sidebar.time_input('End time', value=pd.to_datetime('23:59').time())

# Combine dates and times to create datetime range
start_datetime = pd.to_datetime(f'{start_date} {start_time}')
end_datetime = pd.to_datetime(f'{end_date} {end_time}')

# Numeric range sliders based on min-max values of the variables
side_filter = st.sidebar.selectbox('Select Side', options=df['Side'].unique())
th_filter = st.sidebar.slider('TH Range', min_value=int(df['TH'].min()), max_value=int(df['TH'].max()),
                              value=(int(df['TH'].min()), int(df['TH'].max())))
strength_filter = st.sidebar.slider('Strength Range', min_value=int(df['Strength'].min()),
                                    max_value=int(df['Strength'].max()),
                                    value=(int(df['Strength'].min()), int(df['Strength'].max())))

# Applying numeric filters
filtered_data = df[
    (df['When'] >= start_datetime) & (df['When'] <= end_datetime) &
    (df['Side'] == side_filter) &
    (df['TH'].between(*th_filter)) &
    (df['Strength'].between(*strength_filter))]

# Add a select box for market data indicator selection
market_indicator = st.sidebar.selectbox('Select Market Indicator', [' SPX ', ' NDX ', 'VIX', 'SPY', 'TLT'])

# Additional insights: Display basic statistics of filtered data
st.sidebar.header('Filtered Data Insights')
st.sidebar.write(filtered_data[['Profit_Gross', 'Profit_Net', 'Contracts']].describe())

#########################################################################################################################################################################################################################


## VISUALIZATIONS #######################################################################################################################################################################################################


# Define layout configurations for all figures to maintain consistency
layout_config = {
    'autosize': False,
    'width': 500,  # Set the width to fit two plots side by side neatly
    'height': 400,
    'margin': dict(l=50, r=50, b=100, t=100, pad=4),

}

col1, col2 = st.columns(2)

with col1:
    # 1. Visualization: Cumulative Profit by Date
    filtered_data_sorted = filtered_data.sort_values('When')
    filtered_data_sorted['Cumulative Profit'] = filtered_data_sorted['Profit_Net'].cumsum()
    fig1 = px.line(filtered_data_sorted, x='When', y='Cumulative Profit',
                   title='Cumulative Net Profit Over Time')

    fig1.update_layout(**layout_config)
    st.plotly_chart(fig1)

    # 3. Visualization: Profit Efficiency per Contract
    filtered_data['Profit Per Contract'] = filtered_data['Profit_Net'] / filtered_data['Contracts']
    fig3 = px.bar(filtered_data.groupby('Contracts')['Profit Per Contract'].mean().reset_index(),
                  x='Contracts', y='Profit Per Contract', text='Profit Per Contract')
    fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                       title_text='Average Net Profit per Contract', xaxis_title="Contracts",
                       yaxis_title="Average Profit Per Contract")

    fig3.update_layout(**layout_config)
    st.plotly_chart(fig3)

    # Calculate day of the week from the 'When' column
    filtered_data['DayOfWeek'] = filtered_data['When'].dt.day_name()

    # Aggregate Net Profit by Day of the Week
    profit_by_day = filtered_data.groupby('DayOfWeek')['Profit_Net'].sum().reset_index()

    # Ordering days of the week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    profit_by_day['DayOfWeek'] = pd.Categorical(profit_by_day['DayOfWeek'], categories=days_order, ordered=True)
    profit_by_day = profit_by_day.sort_values('DayOfWeek')

    # Visualization
    fig5 = px.bar(profit_by_day, x='DayOfWeek', y='Profit_Net', title='Net Profit by Day of the Week')
    fig5.update_layout(**layout_config)
    st.plotly_chart(fig5)

with col2:
    # 2. Visualization: Net Profit and Market Indicator Over Time
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    # Adding Profit_Net trace
    fig2.add_trace(go.Scatter(x=filtered_data['When'], y=filtered_data['Profit_Net'], name='Net Profit'),
                   secondary_y=False)
    # Adding dynamic market data trace based on user selection
    fig2.add_trace(go.Scatter(x=filtered_data['When'], y=filtered_data[market_indicator], name=market_indicator,
                              marker_color='lightgrey'), secondary_y=True)
    fig2.update_layout(title_text=f"Net Profit and {market_indicator} Over Time")
    fig2.update_xaxes(title_text="Date")
    fig2.update_yaxes(title_text="<b>Net Profit</b>", secondary_y=False)
    fig2.update_yaxes(title_text=f"<b>{market_indicator}</b>", secondary_y=True, showgrid=False)
    fig2.update_layout(**layout_config)
    st.plotly_chart(fig2)

    # 4.Visualization: Hourly Profit Trends
    filtered_data['Hour'] = filtered_data['When'].dt.hour
    hourly_profit = filtered_data.groupby('Hour')['Profit_Net'].mean().reset_index()
    fig4 = px.line(hourly_profit, x='Hour', y='Profit_Net',
                   title='Average Hourly Profit Trends')
    fig4.update_layout(**layout_config)
    st.plotly_chart(fig4)


    # Assuming 'df' contains multiple market indicators alongside 'Profit_Net'
    correlation_data = df[['Profit_Net', ' SPX ', ' NDX ', 'VIX', 'SPY', 'TLT']].corr()
    fig = px.imshow(correlation_data, text_auto=True, aspect="auto", title="Correlation Heatmap")
    fig.update_layout(**layout_config)
    st.plotly_chart(fig)



