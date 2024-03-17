import time
import requests
import datetime
import streamlit as st
import pandas as pd
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from scipy.stats import norm
import xgboost
from xgboost import XGBRegressor


@dataclass
class data:
    I: object
    E: object


st.set_page_config(
    page_title="Analysis", 
    page_icon="Images/dashboard.png",
    layout = 'wide'

)














class ForecastPlot:
    def __init__(self, data, split):
        self.data = data
        self.split = split

    def plot_forecast(self, original_data, forecast_values, lower_bound, upper_bound,
                      title='Time Series Forecasting', xlabel='Date'):
        fig = go.Figure()

        # Plot original data
        fig.add_trace(go.Scatter(
            x=original_data.index,
            y=original_data,
            mode='lines',
            name='Original Data'
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=original_data.index[-1], periods=len(forecast_values)+1, freq='MS')[1:],
            y=forecast_values,
            mode='lines',
            name='Forecast'
        ))

        # Plot prediction intervals
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=original_data.index[-1], periods=len(lower_bound)+1, freq='MS')[1:],
            y=lower_bound,
            mode='lines',
            name='Lower Bound'
        ))

        fig.add_trace(go.Scatter(
            x=pd.date_range(start=original_data.index[-1], periods=len(upper_bound)+1, freq='MS')[1:],
            y=upper_bound,
            mode='lines',
            name='Upper Bound'
        ))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Values',
            font=dict(size=14, color='#000000'),
            legend=dict(
                orientation='h', 
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                # bgcolor='rgba(0, 0, 0, 0.7)',  
                bordercolor='rgba(255, 255, 255, 0.5)',  
                borderwidth=1,                        
                font=dict(size=12, color='#000000'),   
                
            ),
                height= 600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='#666666', 
                zerolinecolor='#666666'
            ),
            yaxis=dict(
                gridcolor='#666666', 
                zerolinecolor='#666666'  
            )
        )

        return fig
    

class TimeSeriesAnalyzer:
    def __init__(self, data, seasonal=True, m=12, trace=True, suppress_warnings=True):
        self.data = data
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.suppress_warnings = suppress_warnings
        self.best_arima_order = None
        self.best_sarima_order = None

    def find_best_orders(self):
        stepwise_fit = auto_arima(self.data, 
                                  seasonal=self.seasonal,
                                  m=self.m,
                                  trace=self.trace,
                                  suppress_warnings=self.suppress_warnings)

        # Get the best ARIMA and SARIMA orders
        self.best_arima_order = stepwise_fit.get_params()['order']
        self.best_sarima_order = stepwise_fit.get_params()['seasonal_order']

    def display_best_orders(self):
        print("Best ARIMA Order:", self.best_arima_order)
        print("Best SARIMA Order:", self.best_sarima_order)


class SARIMAModel(ForecastPlot):
    
    def __init__(self, data, column, split = 0.8):
        super().__init__(data, split)
        self.rvColumn = column
        self.model = None
        self.results = None
        self.mae_error = None
        self.mse_error = None
        self.r2 = None
        self.rmse = None
        self.mape = None

    def train_model(self, order, seasonal_order):
        self.model = sm.tsa.SARIMAX(self.data[self.rvColumn], order=order, seasonal_order=seasonal_order)
        self.results = self.model.fit(disp=0)


    def test_metrics(self, order, seasonal_order):
        self.tr, self.ts = train_test_split(self.data, train_size=self.split, shuffle=False)
        mdl = sm.tsa.SARIMAX(self.tr[self.rvColumn], order=order, seasonal_order=seasonal_order)
        rs = mdl.fit(disp=0)

        fc = rs.get_forecast(steps=len(self.ts))
        fcv = fc.predicted_mean

        self.mae_error = mean_absolute_error(self.ts, fcv)
        self.mse_error = mean_squared_error(self.ts, fcv)
        self.r2 = r2_score(self.ts, fcv)
        self.rmse = root_mean_squared_error(self.ts, fcv)
        self.mape = mean_absolute_percentage_error(self.ts, fcv)

    def plot_forecast(self, n_months, title='Time Series Forecasting', xlabel='Date'):
        forecast = self.results.get_forecast(steps=n_months)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        
        return super().plot_forecast(self.data[self.rvColumn], forecast_values, 
                              confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1],
                              title=title, xlabel=xlabel)


class ARIMAModel(ForecastPlot):
    
    def __init__(self, data, column, split=0.8):
        super().__init__(data, split)
        self.rvColumn = column
        self.model = None
        self.results = None
        self.mae_error = None
        self.mse_error = None
        self.r2 = None
        self.rmse = None
        self.mape = None

    def train_model(self, order):
        self.model = sm.tsa.ARIMA(self.data[self.rvColumn], order=order)
        self.results = self.model.fit()

    def test_metrics(self, order):
        self.tr, self.ts = train_test_split(self.data, train_size=self.split, shuffle=False)
        mdl = sm.tsa.ARIMA(self.tr[self.rvColumn], order=order)
        rs = mdl.fit()

        fc = rs.get_forecast(steps=len(self.ts))
        fcv = fc.predicted_mean
        self.mae_error = mean_absolute_error(self.ts, fcv)
        self.mse_error = mean_squared_error(self.ts, fcv)
        self.r2 = r2_score(self.ts, fcv)
        self.rmse = root_mean_squared_error(self.ts, fcv)
        self.mape = mean_absolute_percentage_error(self.ts, fcv)

    def plot_forecast(self, n_months, title='Time Series Forecasting', xlabel='Date'):
        forecast = self.results.get_forecast(steps=n_months)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        
        return super().plot_forecast(self.data[self.rvColumn], forecast_values, 
                              confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1],
                              title=title, xlabel=xlabel)
    

import plotly.graph_objects as go

def create_circular_percentage_chart(percentage, title='Circular Percentage Chart'):
    # Create the figure for the circular gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",  # Set the mode to display gauge and number
        value=percentage,  # Set the value (percentage) to be displayed
        title={'text': title, 'font': {'size': 20}},  # Set the title of the chart with larger font size
        gauge={'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "gray", 'ticklen': 10, 'tickfont': {'size': 12}},  # Customize the axis ticks
               'bar': {'color': "#842D78"},  # Set the color of the gauge bar
               'bgcolor': "#DD517F",  # Set the background color of the gauge
               'borderwidth': 2,  # Set the border width
               'bordercolor': "#DD517F",  # Set the border color
               'steps': [
                   {'range': [0, 100], 'color': "#FFB845"}],  # Set the color range for the gauge
               'threshold': {'line': {'color': "#F08D7E", 'width': 4},  # Set the threshold line color and width
                             'thickness': 0.6,  # Set the thickness of the threshold line
                             'value': percentage}}  # Set the threshold value (same as percentage)
    ))

    # Update layout properties for the figure
    fig.update_layout(
        height=300,  # Set the height of the chart
        width=300,  # Set the width of the chart
        margin=dict(t=50, b=50, l=50, r=50),  # Set margins for better spacing
        font={'color': 'black', 'family': 'Arial'},  # Set font color and family
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color
        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background color
    )

    # Show the chart
    return fig





class XGBoostForecast(ForecastPlot):
    def __init__(self, data, targetColumn, lags=5, split_percent=0.8):
        super().__init__(data, split=split_percent)
        self.lags = lags
        self.targetColumn = targetColumn
        self.model = None
        self.mae_error = None
        self.mse_error = None
        self.r2 = None
        self.rmse = None
        self.mape = None
        self.std_error = None
        self.lBound = []
        self.uBound = []

    def series_to_supervised(self, dropnan=True): 
        for i in range(self.lags, 0, -1):
            self.data['lag_'+str(i)] = self.data[self.targetColumn].shift(i)

        if dropnan:
            self.data.dropna(inplace=True)


    def train(self):
        X_train, y_train = self.data.drop(self.targetColumn, axis=1), self.data[self.targetColumn]
        self.model = XGBRegressor()
        self.model.fit(X_train, y_train)

    def test_metrics(self):
        spltData = self.data.copy()
        split_index = int(len(spltData) * self.split)
        train, test = spltData.iloc[:split_index], spltData.iloc[split_index:]

        # Separate features and target variable
        X_train, y_train = train.drop(self.targetColumn, axis=1), train[self.targetColumn]
        X_test, y_test = test.drop(self.targetColumn, axis=1), test[self.targetColumn]

        # Initialize XGBoost regressor, fit and predict
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        self.mae_error = mean_absolute_error(y_test, y_pred)
        self.mse_error = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.rmse = mean_squared_error(y_test, y_pred, squared=False)
        self.mape = mean_absolute_percentage_error(y_test, y_pred)
        # Calculate standard error
        residuals = y_test - y_pred
        self.std_error = np.std(residuals)





    def multi_step_forecast(self, confidence = 0.95, f_horizon=12):
        z = np.round(norm.ppf((1 + confidence) / 2), 2)
        print(z)
        for fh in range(f_horizon):
            next_month = self.data.index[-1] + pd.DateOffset(months=1)
            new_row = pd.Series(index=self.data.columns)
            self.data.loc[next_month] = new_row

            for lag in range(self.lags, 1, -1):
                self.data.loc[self.data.index[-1], 'lag_'+str(lag)] = self.data.iloc[-2]['lag_'+str(lag-1)]
            self.data.loc[self.data.index[-1], 'lag_1'] = self.data.iloc[-2][self.targetColumn]

            self.data.loc[self.data.index[-1], self.targetColumn] = self.model.predict(self.data.iloc[[-1]].drop(self.targetColumn, axis=1))[0]
            self.lBound.append(self.data.loc[self.data.index[-1], self.targetColumn] - z * self.std_error * np.sqrt(fh))
            self.uBound.append(self.data.loc[self.data.index[-1], self.targetColumn] + z * self.std_error * np.sqrt(fh))

    def plot_forecast(self, f_horizon = 12, title='XGBoost Forecasting!', xlabel='Date'):

        forecast_values = self.data[self.targetColumn][-f_horizon:]
        
        return super().plot_forecast(self.data[self.targetColumn][:-f_horizon],
                              forecast_values=forecast_values,
                              title=title,
                              xlabel=xlabel,
                              lower_bound=self.lBound,
                              upper_bound=self.uBound)



        
































if "Imports" not in st.session_state and "Exports" not in st.session_state:
    st.title("Get some data first, Go to Home Page and Download it!")

else:
    d = data(I = st.session_state['Imports'],
            E = st.session_state['Exports'])
    
    rvColImports = 'GEN_VAL_MO'
    rvColExports = 'ALL_VAL_MO'

    filteredDataI = pd.DataFrame(d.I.groupby(['Date']).sum()[rvColImports])
    filteredDataE = pd.DataFrame(d.E.groupby(['Date']).sum()[rvColExports])

    if st.session_state['bestOrdersFlag'] is False:
        with st.spinner('Determining Best orders for Time Series Models, on the current Data! Orders (p, d, q)'):
            tsANA_E = TimeSeriesAnalyzer(filteredDataE[rvColExports])
            tsANA_E.find_best_orders()
            st.session_state['tsANA_E'] = tsANA_E

            tsANA_I = TimeSeriesAnalyzer(filteredDataI[rvColImports])
            tsANA_I.find_best_orders()
            st.session_state['tsANA_I'] = tsANA_I

            st.session_state['bestOrdersFlag'] = True


    ################################################################################################################################################################################################
    tab1, tab2, tab3 = st.tabs(['ARIMA', 'SARIMA', 'XGBOOST'])

    with tab1:

        


        tsE = st.session_state['tsANA_E']
        tsI = st.session_state['tsANA_I']

        c1,c2 = tab1.columns([5,1])

        st.write("")
        monthsAhead = c2.number_input("How far do you wanna look? (in Months)",
                                    key='ma1',
                                    value=12,
                                    format='%d',
                                    placeholder='Lets keep 12 default for now')
        
        arm = ARIMAModel(data=filteredDataE, column=rvColExports)
        arm.train_model(order=tsE.best_arima_order)
        f = arm.plot_forecast(n_months=monthsAhead, title="Forecasting future Exports")
        arm.test_metrics(order=tsE.best_arima_order)
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (arm.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)

        c1,c2 = tab1.columns([5,1])
        arm = ARIMAModel(data=filteredDataI, column=rvColImports)
        arm.train_model(order=tsI.best_arima_order)
        f = arm.plot_forecast(n_months=monthsAhead, title="Forecasting future Imports")
        arm.test_metrics(order=tsI.best_arima_order)
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (arm.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)

    with tab2:
        

        tsE = st.session_state['tsANA_E']
        tsI = st.session_state['tsANA_I']

        c1,c2 = tab2.columns([5,1])

        st.write("")
        monthsAhead = c2.number_input("How far do you wanna look? (in Months)",
                                    key='ma2',
                                    value=12,
                                    format='%d',
                                    placeholder='Lets keep 12 default for now')
        
        arm = SARIMAModel(data=filteredDataE, column=rvColExports)
        arm.train_model(order=tsE.best_arima_order, seasonal_order=tsE.best_sarima_order)
        f = arm.plot_forecast(n_months=monthsAhead, title="Forecasting future Exports")
        arm.test_metrics(order=tsE.best_arima_order, seasonal_order=tsE.best_sarima_order)
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (arm.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)

        c1,c2 = tab2.columns([5,1])
        arm = SARIMAModel(data=filteredDataI, column=rvColImports)
        arm.train_model(order=tsI.best_arima_order, seasonal_order=tsI.best_sarima_order)
        f = arm.plot_forecast(n_months=monthsAhead, title="Forecasting future Imports")
        arm.test_metrics(order=tsI.best_arima_order, seasonal_order=tsI.best_sarima_order)
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (arm.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)



    with tab3:

        c1,c2 = tab3.columns([5,1])
        monthsAhead = c2.number_input("How far do you wanna look? (in Months)",
                                    key='ma3',
                                    value=12,
                                    format='%d',
                                    placeholder='Lets keep 12 default for now')
        
        cf = c2.number_input("How Stringent should your results (Choose High Value for stricter bounds [Wider results])",
                                    key='cf1',
                                    min_value=0.5,
                                    max_value=0.999,
                                    value=0.95,
                                    format='%f',
                                    placeholder='Lets keep 0.95 default for now')
        

        xgE = XGBoostForecast(data=filteredDataE.copy(), targetColumn=rvColExports, lags=5, split_percent=0.8)
        xgE.series_to_supervised(dropnan=True)
        xgE.test_metrics()
        xgE.train()
        xgE.multi_step_forecast(confidence=cf, f_horizon=monthsAhead)
        f = xgE.plot_forecast(f_horizon=monthsAhead, title='Forecasting future Exports')
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (xgE.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)

        c1,c2 = tab3.columns([5,1])
        xgI = XGBoostForecast(data=filteredDataI.copy(), targetColumn=rvColImports, lags=5, split_percent=0.8)
        xgI.series_to_supervised(dropnan=True)
        xgI.test_metrics()
        xgI.train()
        xgI.multi_step_forecast(confidence=cf, f_horizon=monthsAhead)
        f = xgI.plot_forecast(f_horizon=monthsAhead, title='Forecasting future Imports')
        c1.plotly_chart(f, use_container_width=True)
        p = create_circular_percentage_chart(100 - (xgI.mape*100), title='How good is the model')
        c2.plotly_chart(p, use_container_width=True)

