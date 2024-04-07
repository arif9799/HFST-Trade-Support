import time
import requests
import datetime
import streamlit as st
import pandas as pd
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


@dataclass
class data:
    I: object
    E: object

class SankeyTradeAnalysis:
    def __init__(self, k=3, n=10):
        self.k = k
        self.n = n

    def compute_top_targets(self, dataFrame, country, source, destination, values):
        topk_countries = (dataFrame.groupby(country)[values].sum()
                          .nlargest(self.k).index)
        
        topk_data = dataFrame[dataFrame[country].isin(topk_countries)]
        
        result = (topk_data.groupby([source, destination])[values].sum()
                  .reset_index()
                  .groupby(country)
                  .apply(lambda x: x.nlargest(self.n, values))
                  .reset_index(drop=True))
        
        result.columns = ['source', 'target', 'values']
        result = result[result['values'] != 0]
        result['values'] = round(result['values'] / 1e9, 4)
        return result

    def map_nodes_and_create_links_dataframe(self, *dataframes):
        links = pd.concat(dataframes, axis=0)
        uniqueNodes = list(pd.unique(links[['source', 'target']].values.ravel('k')))
        mappedNodes = {k: v for v, k in enumerate(uniqueNodes)}
        links['source'] = links['source'].map(mappedNodes)
        links['target'] = links['target'].map(mappedNodes)
        linksDict = links.to_dict(orient='list')
        return uniqueNodes, linksDict

    def get_node_color(self, label):
        if label.endswith('(i)'):
            return "#a796e8"
        elif label.endswith('(e)'):
            return "#66023c"
        else:
            return "#66786c"

    def plot_sankey_diagram(self, uniqueNodes, linksDict, title):
        fig = go.Figure(data=[
            go.Sankey(
                    node=dict(
                        pad=10,
                        thickness=30,
                        line=dict(color="black", width=0.5),
                        label=uniqueNodes,
                        color=[self.get_node_color(label) for label in uniqueNodes],
                        line_width=0,
                    ),
                    link=dict(
                        source=linksDict['source'],
                        target=linksDict['target'],
                        value=linksDict['values'],
                        color="#ffcea5"
                    )
                ),
            
            #Custom Legend
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Top Countries by Valuation that US Imported from",
                marker=dict(size=21, color="#a796e8", symbol='circle'),
            ),

            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Top Countries by Valuation that US Exported to",
                marker=dict(size=21, color="#66023c", symbol='circle'),
            ),

            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="The US States, where International Trade Transactions occured",
                marker=dict(size=21, color="#66786c", symbol='circle'),
            )
        ])
        
        fig.update_layout(
            legend=dict(
                orientation="h", 
                y=-0.1,  
                x=0.5,   
                xanchor='center', 
                yanchor='top',    
                bgcolor='rgba(255, 255, 255, 0.7)', 
                bordercolor='rgba(0, 0, 0, 0.5)',      
                borderwidth=1,                         
            ),
            title_text=title,
            font_size=10,
            title_font_size=20,
            height=950,
            width=1000,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
            # paper_bgcolor='#c9ffc7',
            # plot_bgcolor='#c9ffc7'
        )


        # Remove axis values
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig

    def performSankey(self, importDF, exportDF, title='default'):

        importsCTYtoUSDT = self.compute_top_targets(dataFrame=importDF,
                                                              country='CTYi',
                                                              source='CTYi',
                                                              destination='US_STATE',
                                                              values='GEN_VAL_MO')

        exportsUSDTtoCTY = self.compute_top_targets(dataFrame=exportDF,
                                                              country='CTYe',
                                                              source='US_STATE',
                                                              destination='CTYe',
                                                              values='ALL_VAL_MO')

        uniqueNodes, linksDict = self.map_nodes_and_create_links_dataframe(importsCTYtoUSDT, 
                                                                                     exportsUSDTtoCTY)

        return self.plot_sankey_diagram(uniqueNodes=uniqueNodes, 
                                           linksDict=linksDict, 
                                           title=title)

class DonutChartCreator:
    def __init__(self, data, value, loc):
        self.data = data
        self.value = value
        self.loc = loc

    def create_donut_chart(self, k=3, hole=0.6, title_suffix=""):
        df = self.data

        # Calculate import/export valuations for each location
        location_valuations = df.groupby(self.loc)[self.value].sum().reset_index()

        # Sort locations based on valuations
        sorted_locations = location_valuations.sort_values(by=self.value, ascending=False)

        # Select top k locations and sum the remaining valuations
        top_k_locations = sorted_locations.head(k)
        others_valuation = sorted_locations.iloc[k:][self.value].sum()

        # Create a new DataFrame with top k locations and the "others" category
        top_locations_and_others = pd.concat([top_k_locations,
                                              pd.DataFrame({self.loc: ['Others'], self.value: [others_valuation]})])

        # Create a donut chart
        fig = px.pie(top_locations_and_others, values=self.value, names=self.loc, hole=hole,
                     title=f'Top {k} {title_suffix}')

        # Customizing layout
        fig.update_traces(
            textinfo='percent+label',
            marker=dict(line=dict(color='#000000', width=2)),
            pull=[0.05] * (k + 1)  # Pulling slices for emphasis
        )
        
        # Update layout
        fig.update_layout(
            title=dict(font=dict(size=24, color='#000000')),  
            font=dict(size=14, color='#333333'), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # fig.show("svg")
        return fig

class TimeSeriesVisualizer:
    def __init__(self):
        pass
        
    def visualize_time_series(self, x, ys, xlabel, ylabels, title=''):
        fig = go.Figure()


        for y, label in zip(ys, ylabels):
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=label))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Values',
            font=dict(size=14, color='#ffffff'),
            legend=dict(
                orientation='h', 
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0, 0, 0, 0.7)',  
                bordercolor='rgba(255, 255, 255, 0.5)',  
                borderwidth=1,                        
                font=dict(size=12, color='#ffffff'),   
                
            ),
                height= 600,
                width = 1000,
            # plot_bgcolor='#1f1f1f', 
            # paper_bgcolor='#1f1f1f', 
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

st.set_page_config(
    page_title="Analysis", 
    page_icon="Images/dashboard.png",
    layout = 'wide')

if "Imports" not in st.session_state and "Exports" not in st.session_state:
    st.title("Get some data first! Go to Home Page and Download it!")

else:
    d = data(I = st.session_state['Imports'], E = st.session_state['Exports'])

    ################################################################################################################################################################################################
    tab1, tab2, tab3 = st.tabs(['Sankey Charts', 'Donut Charts', 'Time Series Charts'])

    with tab1:
        c1,c2 = tab1.columns([1,1])
        topk = c1.selectbox(label="Top K countries to take into consideration",
                            options=[i+1 for i in range(15)],
                            index=2,
                            key='c1tab1'
                            )
        topn = c2.selectbox(label="Top N US states to take into consideration", 
                            options=[i+1 for i in range(15)],
                            index=4,
                            key='c2tab1'
                            )
        trade_analysis = SankeyTradeAnalysis(k=int(topk),
                                        n=int(topn))
        f = trade_analysis.performSankey(importDF=d.I, 
                                exportDF=d.E, 
                                title=d.I.at[0,'I_COMMODITY_SDESC'] + ' US Imports/Exports (2013-2023) in billions (USD)')
        
        st.header("Sankey Chart")
        st.plotly_chart(f, use_container_width=True)

    with tab2:
        c1,c2 = tab2.columns([1,1])
        topk = c1.selectbox(label="Top K countries to take into consideration",
                            options=[i+1 for i in range(15)],
                            index=2,
                            key='c1tab2'
                            )
        topn = c2.selectbox(label="Top N US states to take into consideration", 
                            options=[i+1 for i in range(15)],
                            index=4,
                            key='c2tab2'
                            )
        
        # Top 5 Countries US imports from
        dCC = DonutChartCreator(data = d.I, value='GEN_VAL_MO', loc='CTY_NAME')
        d1 = dCC.create_donut_chart(k=topk, title_suffix=f"Countries US Imports from, {d.I.at[0,'I_COMMODITY']} <br> {d.I.at[0,'I_COMMODITY_SDESC']}")

        # Top 5 countries US exports to
        dCC = DonutChartCreator(data = d.E, value='ALL_VAL_MO', loc='CTY_NAME')
        d2 = dCC.create_donut_chart(k=topk, title_suffix=f"Countries US Exports to, {d.E.at[0,'E_COMMODITY']} <br> {d.E.at[0,'E_COMMODITY_SDESC']}")

        # Top 5 US States importers
        dCC = DonutChartCreator(data = d.I, value='GEN_VAL_MO', loc='US_STATE')
        d3 = dCC.create_donut_chart(k=topn, title_suffix=f"Importer States of USA, for HS {d.I.at[0,'I_COMMODITY']} <br> {d.I.at[0,'I_COMMODITY_SDESC']}")

        # Top 5 US States exporters
        dCC = DonutChartCreator(data = d.E, value='ALL_VAL_MO', loc='US_STATE')
        d4 = dCC.create_donut_chart(k=topn, title_suffix=f"Exporter States of USA, for HS {d.E.at[0,'E_COMMODITY']} <br> {d.E.at[0,'E_COMMODITY_SDESC']}")

        c1,c2, c3 = tab2.columns([2,0.5,2])
        c1.plotly_chart(d1)
        c1.plotly_chart(d2)
        c3.plotly_chart(d3)
        c3.plotly_chart(d4)

    with tab3:

        sDate = min(d.I.Date)
        eDate = max(d.I.Date)
        fImports = d.I.groupby(['Date'])['GEN_VAL_MO'].sum().reset_index()
        fExports = d.E.groupby(['Date'])['ALL_VAL_MO'].sum().reset_index()
        chartTitle = f'US International Trade, Time Series Chart'      

        # Create an instance of TimeSeriesVisualizer for Imports
        visualizer = TimeSeriesVisualizer()

        # # Visualize time series graph
        ts = visualizer.visualize_time_series(x = fImports.Date,
                                        ys = [fImports.GEN_VAL_MO, fExports.ALL_VAL_MO],
                                        xlabel = 'Date',
                                        ylabels = ['Imports', 'Exports'],
                                        title=chartTitle)
        tab3.write(chartTitle)
        tab3.plotly_chart(ts, use_container_width=True)

