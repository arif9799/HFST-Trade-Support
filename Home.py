import time
import requests
import datetime
import streamlit as st
import pandas as pd
from dataclasses import dataclass
import numpy as np

st.set_page_config(
    page_title="Commodities Trading", 
    page_icon="Images/chart.png",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
            ## Commodities Trading with AI
            
            **GitHub**: Github Link
    
            This AI Webapp is meant to let you source data on the fly
            and perform Visual Analytics, Time Series Forecast and Risk
            Analysis of the Predictions
        """
    }
)


@dataclass
class data:
    host: str 
    dataHost: str 
    api: str
    im: str
    ex: str
    clsf: str


class DataRetriever:
    @staticmethod
    def retrieve_data(url, predicates):
        """
        Static method to retrieve data from the provided URL with specified predicates.

        Parameters:
            url (str): The URL to retrieve data from.
            predicates (dict): The predicates to be included in the request.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the retrieved data, or None if an error occurs.
        """
        RESULTS = requests.get(url, params=predicates)

        if RESULTS.status_code == 200:
            DATA = RESULTS.json()
            DF = pd.DataFrame(columns=DATA[0], data=DATA[1:])
        else:
            print("Error:", RESULTS.status_code)
            DF = None
    
        return DF, RESULTS


class TradeDataFilter:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    @staticmethod
    def extract_state(dist_name):
        parts = dist_name.split(',')
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            return dist_name

    def filter_data(self, **kwargs):
        filtered_data = self.data_frame.copy()

        for key, value in kwargs.items():
            if key == 'start_date':
                filtered_data = filtered_data[filtered_data['Date'] >= value]
            elif key == 'end_date':
                filtered_data = filtered_data[filtered_data['Date'] <= value]
            elif key in filtered_data.columns:
                if isinstance(value, list):
                    # Handle list values (e.g., for multiple commodities, countries, etc.)
                    filtered_data = filtered_data[filtered_data[key].isin(value)]
                else:
                    # Handle single value
                    filtered_data = filtered_data[filtered_data[key] == value]
            else:
                print(f"Warning: '{key}' column not found in data. Skipping filtering by this condition.")

        if filtered_data.empty:
            print("Error: Filtered data is empty.")
            return None
        else:
            return filtered_data.reset_index(drop=True)



d = data(
            host = "https://api.census.gov/data",
            dataHost= "timeseries/intltrade",
            api = "919d92868c70f630a9eda2dc7445e4ad1f614ab6",
            im='imports',
            ex='exports',
            clsf='hs'
         )

importUrl =  "/".join([d.host, d.dataHost, d.im, d.clsf] )
exportUrl =  "/".join([d.host, d.dataHost, d.ex, d.clsf] )




st.sidebar.title('What all you can do!')

st.title("⌂ Commodities Trading with AI!")
st.write("Download some data here!")
st.text("Monthly International Trade Time Series Data!")


#-------------------------------------------- FORM IS HERE ----------------------------------------------------------------
with st.form(key = "parameters"):
    st.write("Please select appropriate options to fetch desired data, the options will be constructed into RestAPI!!")
    st.text("""Based on the parameters you select, we'll extract data on the fly from US 
Census Bureau Website. It provides International Monthly Trade Time Series 
Data. We'll fetch imports and exports valuations for the HS Code you ask 
for, within the timeframe you specified""")

    
    #-------------------------------------------- HS CODE INPUT ----------------------------------------------------------------
    st.write("")
    commCode = st.number_input("Insert Harmonized System (HS) Code ",
                                    key='commCode',
                                    value=27,
                                    format='%d',
                                    placeholder='Lets keep 27 default for now',
                                    help='It could be 2 digit Chapter Number, 4 digit Heading number, 6 digit Sub-Heading or 10 digit exact Product ID')


    #-------------------------------------------- HS LEVEL INPUT ----------------------------------------------------------------
    st.write("")
    hsOptions = {"HS2 (if you enetered 2 digit HS Code): Chapter Number": "HS2",
                "HS4 (if you enetered 4 digit HS Code): Heading ": "HS4",
                "HS6 (if you enetered 6 digit HS Code): Sub-Heading": "HS6",
                "HS10 (if you enetered 10 digit HS Code): Product-ID": "HS10"}

    commLevel = st.selectbox(
    "Select HS Level",
    options=hsOptions.keys(),
    index=None,
    placeholder="Select HS Level...",
    key='commLevel'
    )

    #-------------------------------------------- INDIVIDUAL/ GROUPED COUNTRIES LEVEL ----------------------------------------------------------------
    st.write("")
    smOptions = {"DET (Summarization of Individual Country Import/Export Values)": "DET",
                "CGP (Summarization of Grouped Countries Import/Export Values, example European Union)": "CGP"}
    smLevel = st.selectbox(
    "Select Summarization",
    options=smOptions.keys(),
    index=None,
    placeholder="Select Summary Level...",
    key='smLevel'
    )


    #-------------------------------------------- DATE RANGE ----------------------------------------------------------------
    st.write("")
    st.write("Enter the range of dates to fetch data for")
    st.text("US Census Bureau provides Monthly Interational Trade Data from 2013 onwards")
    col1, col2 = st.columns([1,1])
    startYear = col1.selectbox('Start Year', range(2013, datetime.datetime.now().year + 1))
    endYear = col2.selectbox('End Year', range(2013, datetime.datetime.now().year + 1))


    # Every form must have a submit button.
    st.write("")
    st.write("")
    submitted = st.form_submit_button("Download")




dr = DataRetriever()
if submitted:
    if endYear < startYear:
        st.error('End Year Cannot be before Start Year')
    else:
        with st.spinner('Fetching the Data, it may take several minutes'):
            st.session_state['bestOrdersFlag'] = False
            VARIABLES = [
                        'I_COMMODITY_SDESC',
                        'DISTRICT',
                        'DIST_NAME',
                        'CTY_CODE',
                        'CTY_NAME',
                        'GEN_VAL_MO', 
                        'CON_VAL_MO'
                        ]

            PREDICATES = {}
            PREDICATES['get'] = ",".join(VARIABLES)
            PREDICATES['YEAR'] = [str(i) for i in range(startYear,endYear+1)]
            PREDICATES['MONTH'] = [str(i).zfill(2) for i in range(1,13)]
            PREDICATES['I_COMMODITY'] = str(commCode).zfill(2)
            PREDICATES['COMM_LVL'] = str(hsOptions[commLevel])
            PREDICATES['SUMMARY_LVL'] = str(smOptions[smLevel])
            PREDICATES['SUMMARY_LVL2'] = 'HSCYDT'

            ImportsData, _ = dr.retrieve_data(url= importUrl, predicates=PREDICATES)


            VARIABLES = [
                        'E_COMMODITY_SDESC',
                        'DISTRICT',
                        'DIST_NAME',
                        'CTY_CODE',
                        'CTY_NAME',
                        'ALL_VAL_MO'
                        ]

            PREDICATES = {}
            PREDICATES['get'] = ",".join(VARIABLES)
            PREDICATES['YEAR'] = [str(i) for i in range(startYear,endYear+1)]
            PREDICATES['MONTH'] = [str(i).zfill(2) for i in range(1,13)]
            PREDICATES['E_COMMODITY'] = str(commCode).zfill(2)
            PREDICATES['COMM_LVL'] = str(hsOptions[commLevel])
            PREDICATES['SUMMARY_LVL'] = str(smOptions[smLevel])
            PREDICATES['SUMMARY_LVL2'] = 'HSDTCY'

            ExportsData, _ = dr.retrieve_data(url= exportUrl, predicates=PREDICATES)

        
        ectypes = {
            'E_COMMODITY_SDESC' : str,
            'COMM_LVL' : str,
            'DISTRICT': int,
            'DIST_NAME': str,
            'CTY_CODE': int,
            'CTY_NAME': str,
            'ALL_VAL_MO':np.int64,
            'YEAR': int,
            'MONTH':int,
            'E_COMMODITY':np.int64,
            'SUMMARY_LVL': str,
            'SUMMARY_LVL2': str
        }

        ictypes = {
            'I_COMMODITY_SDESC' : str,
            'COMM_LVL' : str,
            'DISTRICT': int,
            'DIST_NAME': str,
            'CTY_CODE': int,
            'CTY_NAME': str,
            'GEN_VAL_MO':np.int64,
            'CON_VAL_MO':np.int64,
            'YEAR': int,
            'MONTH':int,
            'I_COMMODITY':np.int64,
            'SUMMARY_LVL': str,
            'SUMMARY_LVL2': str
        }

        # Convert data types accordingly
        ExportsData = ExportsData.astype(ectypes)

        # Convert data types accordingly
        ImportsData = ImportsData.astype(ictypes)

        ImportsData['US_STATE'] = ImportsData.DIST_NAME.apply(TradeDataFilter.extract_state)
        ExportsData['US_STATE'] = ExportsData.DIST_NAME.apply(TradeDataFilter.extract_state)

        ImportsData['CTYi'] = ImportsData.CTY_NAME.apply(lambda x: x + '(i)')
        ExportsData['CTYe'] = ExportsData.CTY_NAME.apply(lambda x: x + '(e)')

        ImportsData['Date'] = pd.to_datetime(ImportsData.YEAR.astype(str) + '-' + ImportsData.MONTH.astype(str), format='%Y-%m')
        ExportsData['Date'] = pd.to_datetime(ExportsData.YEAR.astype(str) + '-' + ExportsData.MONTH.astype(str), format='%Y-%m')

        
        st.success("Data Retrieved, you can proceed for Analysis, and you can take a copy as well, by clicking Dwonload as CSV button on the top right corner of the table")
        st.session_state['Imports'] = ImportsData
        st.session_state['Exports'] = ExportsData
        st.write("Imports Data")
        st.dataframe(ImportsData, use_container_width=True)
        st.write("Exports Data")
        st.dataframe(ExportsData, use_container_width=True)
