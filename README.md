# GUI SETUP

To access the streamlit web app, download/pull the contents in [The GUI Folder](TTGDataAnalysis/tree/main/GUI).

We tried experimenting with Figma to create a custom UI, however we loved the interactive Plotly graphs on the dashboard. These graphs are very simple to code and give the option to quickly create new ones at your leisure. The alternative was to create our own version of Plotly, which would potentially waste time on your end and make the dashboard harder to configure.

## Initial Setup

Once the GUI folder is open, install the following packages:

`pip install streamlit`
`pip install pickle`
`pip install plotly`
`pip install pandas`
`pip install openai` (Optional)

To access the AI assistant, you should create an OpenAI account to generate an API key and add a small amount of funds to specify a model and use tokens. Refer to the [OpenAI API Documentation](https://platform.openai.com/docs/overview). 

Once the OpenAI stuff is set, generate an API key and replace the empty string in **line 21** of [TTG_Insights_Dash.py](TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py).

If you do not want to access the AI assistant and just see the dashboard comment out the code from lines 20-54 in [TTG_Insights_Dash.py](TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py).

1. **For demo**
   After installing all the necessary packages, having [TTG_Insights_Dash.py](TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py) and [my_dataframe.pkl](TTGDataAnalysis/blob/main/GUI/my_dataframe.pkl) open, run 'streamlit run TTG_Insights_Dash.py' in your terminal/command prompt (make sure you are in the right directory first). This will show the dashboard based on the initial trade data you gave us.

2. **For actual usage**
   Refer to [DataPipeline.ipynb](TTGDataAnalysis/blob/main/GUI/DataPipeline.ipynb). This file creates the .pkl file (Pandas df) which is used in the GUI. Configure it to pull the most recent data from the SQL DB.
