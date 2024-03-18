<h2>
    
**_Project Description:_**
</h2>

- This Project provides a Web Application, a peculiar front end that lets a non-technical user, who is interested in Commodities Trading, Analyse Imports and Exports of any Commodity on the go!
- The data is accesible via RestAPIs, retrieving Imports/Exports information or Monthly Internation Trade data of United States via US Census Bureau Website
- Providing the HS Code for Commodity of Interest, you can download data on the fly and Perform Analysis as well as Forecasting right off the bat! 
- The Web Application is hosted on Streamlit Community Cloud, here's the link () but it is very very very slow [Will soon figure a way to deploy on a faster platform]
- I would recommend running it locally for effective results and smooth experience (instructions to do so, are given below)

<br/>
<br/>
<br/>
<br/>


<h2>
    
**_How to run Locally:_**
</h2>

- Its as simple as running any other application on your local system
- Download/Fork/Clone this repository on to your local system
- ```Home.py``` file is the main application that needs to be fired up
- Run the following command in the directory where ```Home.py``` application resides   
      - ```console
          Commodity-Web-Application (directory) % streamlit run Home.py
      ```
- The above system step will pop open localhost and you'll have access to the application

<br/>
<br/>
<br/>
<br/>



<h2>
    
**_How to navigate Web Application:_**
</h2>



<br/>
<h3> 

**_Starting off by downloading the data first:_**
</h3>

- The following steps will show you how to navigate and make the best use of the Commodities Trading Web Application
- When you fire up the Web Application, you'll land on Home Page where you can specify details of the commodity to download its data
- You have to mention the exact 2,4,6 or 10 digit HS Code along with HS Level and Summarization Level and timeframe for which you want the data
- Refer the image Below for better understanding of how to download the data

<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/HomePage1.png" width="1000" alt="Description">
    </div>
<br/>

<br/>
<br/>
- Once the data has been downloaded, you can preview it in the same tab (and even download it as csv)
- Preview of a successful retrieval of data on the fly is as follows  

<br/>
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/HomePage2.png" width="1000" alt="Description">
    </div>
<br/>



<br/>

<h3> 

**_Performing Visual Analysis:_**
</h3>

- Once you have the data, switch to Visual Analysis Page for performing intutive and visual inspection of the data
- You have Sankey charts to view the flow of Imports into the states and exports out of the states
- You can also tweak top k countries (trade partners of United States) and top n states in USA that execute International Trade
- Refer Image Below:
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/VisPage1.png" width="1000" alt="Description">
    </div>
<br/>

- Up Next, You have Donut charts to view the top trading partners of United States and their share of Contributions in Imports/ Exports
- Here as well, You can also tweak top k countries (trade partners of United States) and top n states in USA that execute International Trade
- Refer Image Below:
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/VisPage2.png" width="1000" alt="Description">
    </div>
<br/>

- Finally, You have Time Series charts to view the trends (increasing/decreasing) and patterns (Seasonal/Random) in Imports/ Exports
- All the Visuals on this Web Applications are Interactive Charts, you can hover to view Valuations of Imports/Exports
- Refer Image Below:
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/VisPage3.png" width="1000" alt="Description">
    </div>
<br/>




<br/>

<h3> 

**_Commodity Valuations Forecasting:_**
</h3>

- You can use ARIMA, SARIMA or XGBOOST Algrithms to predict Valuations of Imports/Exports in the near future
- You can view the time series data and the forecasts along with prediction intervals on the screen
- You can also dynamically choose the forecasting horizon (how far would you wanna look into future)?
- You also have guage meters on the right showcasing how well the models perform based on MAPE
- You can also specify the confidence level for prediction intervals based on the stringency of your application
- You can refer the 3 images below (pertaining to ARIMA, SARIMA and XGBOOST)

<br/>
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/fcPage1.png" width="1000" alt="Description">
    </div>
<br/>

<br/>
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/fcPage2.png" width="1000" alt="Description">
    </div>
<br/>

<br/>
<br/>
    <div align="center">
      <img src="https://github.com/arif9799/Commodity-Web-Application/blob/main/images/fcPage3.png" width="1000" alt="Description">
    </div>
<br/>

