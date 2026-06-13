# AeroRight — Operational Risk and Delay Management in U.S. Airlines

AeroRight is a data-driven web application built on 10 years of U.S. airline data to help passengers and airline operators make smarter, risk-aware travel decisions.

## Live App
https://airline-delay-tool-ezd765kyfxxyruo3dkjxhh.streamlit.app

## About
AeroRight analyzes 224,634 records across 21 airlines and 300 airports from the U.S. Bureau of Transportation Statistics (BTS) between 2015 and 2025. The app uses a trained XGBoost model with an AUC of 0.89 and accuracy of 83% to identify cancellation risk patterns based on historical data.

## Tools

**Customer Tool**
Designed for passengers. Enter an airline, airport, and month to check the historical cancellation risk and get a safer airline recommendation.

**Airline Tool**
Designed for airline operations managers. Select an airline and month to see a full breakdown of delay causes and identify the biggest operational risk driver.

**Recommendation System**
Designed for passengers and travel agencies. Select an airport and month to get the top 3 safest airlines ranked by cancellation risk.

## Key Findings
Carrier operations account for 70 to 80 percent of all delay minutes across every major U.S. airport. Delta Air Lines had the lowest cancellation rate of 0.1 percent across the full 10 year period. April is the most dangerous month to fly with a 3.5 percent cancellation rate. The year 2025 recorded the highest total delay minutes at 124.7 million.

## Tech Stack
Python, XGBoost, Streamlit, Pandas, Scikit-learn, BTS Data

## Built By
Sai Charan Raju Bollepalli
Business Analytics Capstone, Group 4
