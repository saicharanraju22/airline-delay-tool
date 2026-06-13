# AeroRight — Operational Risk and Delay Management in U.S. Airlines

AeroRight is a data-driven web application built on 10 years of U.S. airline data to help passengers and airline operators make smarter, risk-aware travel decisions.

## Live App
https://airline-delay-tool-ezd765kyfxxyruo3dkjxhh.streamlit.app

## About
AeroRight analyzes 224,634 records across 21 airlines and 300 airports from the U.S. Bureau of Transportation Statistics (BTS) between 2015 and 2025. The app uses a trained XGBoost model with an AUC of 0.89 and accuracy of 83 percent to identify cancellation risk patterns based on historical data.

## Analysis

This project followed the CRISP-DM framework across six phases — business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

**Business Understanding**
Five research questions were defined to address why U.S. airline delays repeat, which carriers and airports consistently underperform, and when cancellation risk is highest — problems the industry currently handles reactively rather than proactively.

**Data Understanding**
224,634 records were explored across 21 airlines, 300 airports, and 10 years. Key findings included carrier operations accounting for 70 to 80 percent of all delay minutes, summer peaks in June and July with delays double those of September and October, and 2025 recording the highest total delay minutes at 124.7 million.

**Data Preparation**
Missing values were handled by dropping 354 rows with no operational meaning and imputing 258 rows with zero. Log transformation was applied to stabilize skewed delay distributions. Outliers were capped at the 99.9th percentile using winsorization. Class imbalance of 75 percent low risk versus 25 percent high risk was addressed using scale_pos_weight.

**Modeling**
Multicollinearity was tested using VIF and carrier delay was removed at VIF 8.3. High correlation between carrier delay and late aircraft at 0.89 was resolved by engineering three ratio features. Three models were built and compared — Logistic Regression, Random Forest, and XGBoost.

**Evaluation**
XGBoost was selected as the best model with 83 percent accuracy, 78 percent recall, 64 percent precision, F1 score of 70 percent, and AUC of 0.89. Out-of-time validation was used on 2024 and 2025 holdout data to confirm the model generalizes to unseen real-world conditions.

**Deployment**
The analytical output was deployed as AeroRight — a three tool Streamlit web application accessible to passengers, airline managers, and travel agencies without any coding knowledge required.

## Tools

**Customer Tool**
Designed for passengers. Enter an airport, airline, and month to check the historical cancellation risk and get a safer airline recommendation.

**Airline Tool**
Designed for airline operations managers. Select an airline and month to see a full breakdown of delay causes and identify the biggest operational risk driver.

**Recommendation System**
Designed for passengers and travel agencies. Select an airport and month to get the top 3 safest airlines ranked by cancellation risk.

## Key Findings
Carrier operations account for 70 to 80 percent of all delay minutes across every major U.S. airport. Delta Air Lines had the lowest cancellation rate of 0.1 percent across the full 10 year period. April is the most dangerous month to fly with a 3.5 percent cancellation rate. The year 2025 recorded the highest total delay minutes at 124.7 million.

## Tech Stack
Python, XGBoost, Streamlit, Pandas, Scikit-learn, BTS Data

## Limitations
- Data is aggregated at the monthly level by carrier and airport — individual flight level risk cannot be assessed
- All predictions are based on historical patterns from 2015 to 2025 and do not reflect real-time conditions
- Weather events, air traffic control disruptions, and airline operational changes happening today are not captured

## Future Work
- Year-specific delay risk forecasting so users can see how risk shifts year over year
- Live BTS data feed so the app updates automatically every month without manual intervention
- Individual flight level data so passengers can check risk for a specific flight number and departure time
- Integration with airline booking platforms so passengers see cancellation risk at the point of booking

## Built By
Sai Charan Raju Bollepalli
Business Analytics Capstone, Group 4
