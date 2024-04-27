# Info
Tartigrade has given us a complex series of trades involving Stocks, options, futures contracts and FX (mainly CAD.USD).

They have greatly simplified the definition of their trades by building the following Key Trade Characteristics:
![Screenshot 2024-04-27 at 3 21 53 PM](https://github.com/ammaarmelethil/TTGDataAnalysis/assets/100314064/8ac66d12-8e94-40ed-8852-66496e978fdd)

We have developed the following dashboard allowing them to visualize live and past data by filtering through controllable features.
![ScreenRecording2024-04-11at8 40 30PM-ezgif com-video-to-gif-converter](https://github.com/ammaarmelethil/TTGDataAnalysis/assets/100314064/9ac2e22e-1da2-4281-98f9-a23beb37f5a8)

And a 24/7 AI analyst powered by OpenAI to answer all questions regarding the data in their DB
<img width="793" alt="Screenshot 2024-04-10 at 5 09 06 PM" src="https://github.com/ammaarmelethil/TTGDataAnalysis/assets/100314064/04d17b04-240a-4f7b-94f5-7f7e2786987d">

<img width="737" alt="Screenshot 2024-04-10 at 5 30 02 PM" src="https://github.com/ammaarmelethil/TTGDataAnalysis/assets/100314064/2393d1f9-ef57-4f27-861f-7ca7f5e0e31e">

## Initial Setup

Once the GUI folder is open, install the following packages:

`pip install streamlit`
`pip install pickle`
`pip install plotly`
`pip install pandas`
`pip install openai` (OpenAI is optional)

To access the AI assistant, you should create an OpenAI account to generate an API key and add a small amount of funds to specify a model and use tokens. Refer to the [OpenAI API Documentation](https://platform.openai.com/docs/overview). 

Once the OpenAI stuff is set, generate an API key and replace the empty string in **line 21** of [TTG_Insights_Dash.py](https://github.com/ammaarmelethil/TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py).

If you do not want to access the AI assistant and just see the dashboard comment out the code from lines 20-54 in [TTG_Insights_Dash.py](https://github.com/ammaarmelethil/TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py).

1. **For demo**
   After installing all the necessary packages, having [TTG_Insights_Dash.py](https://github.com/ammaarmelethil/TTGDataAnalysis/blob/main/GUI/TTG_Insights_Dash.py) and [my_dataframe.pkl](https://github.com/ammaarmelethil/TTGDataAnalysis/blob/main/GUI/my_dataframe.pkl) open, run 'streamlit run TTG_Insights_Dash.py' in your terminal/command prompt (make sure you are in the right directory first). This will show the dashboard based on the initial trade data you gave us.

2. **For actual usage**
   Refer to [DataPipeline.ipynb](https://github.com/ammaarmelethil/TTGDataAnalysis/blob/main/GUI/DataPipeline.ipynb). This file creates the .pkl file (Pandas df) which is used in the GUI. Configure it to pull the most recent data from the SQL DB.
