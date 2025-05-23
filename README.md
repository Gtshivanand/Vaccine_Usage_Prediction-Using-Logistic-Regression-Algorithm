# 🛡️💉 Vaccine Usage Prediction - Using Logistic Regression Algorithm 📊📈

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)
](https://www.anaconda.com)

<img src="https://github.com/Gtshivanand/Vaccine_Usage_Prediction-Using-Logistic-Regression-Algorithm/blob/main/images/Vaccine%20Usage%20Prediction.jpg"/>

## Introduction:
Vaccination plays a crucial role in preventing the spread of infectious diseases and ensuring public health. However, vaccine hesitancy remains a major challenge, influenced by various demographic, psychological, and socio-economic factors.

This project, "Vaccine Usage Prediction - Using Logistic Regression Algorithm", aims to predict whether an individual is likely to take a vaccine based on different influencing factors such as age, health conditions, income level, and trust in medical institutions.

By leveraging Logistic Regression, a widely used classification algorithm, we can analyze trends, identify key factors affecting vaccine acceptance, and provide valuable insights for policymakers and healthcare professionals. The results of this project can be used to design effective awareness campaigns and improve vaccination rates globally.


## Problem Statement:

The data set is the response of people to the h1n1 flu vaccine related questionnaire. The respondents are people of age 6 months and older. This survey was designed to monitor the influenza immunization coverage in 2009-10 season. Machine learning techniques may aid a more efficient analysis in the prediction of how likely the people are to opt for the flu vaccine. In this case study, we predict, how likely it is that the people will take a H1N1 flu vaccine.        


## Project Overview:
This project leverages machine learning techniques to analyze and predict vaccine usage based on key demographic and behavioral features. By utilizing **Logistic Regression**, the model determines the likelihood of individuals opting for vaccination based on historical and survey data.


## Dataset Information:
The dataset used in this project contains survey responses from individuals regarding their decision to take a vaccine. It includes demographic, health-related, and social factors affecting vaccination behavior. The dataset consists of features such as **age, gender, education level, occupation, health conditions, and vaccination history**.

- Filename: h1n1_vaccine_prediction.csv
- Number of Rows: 26,707
- Number of Columns: 34
- Dataset Type: Mixed (Numerical & Categorical)
- Target Variable: h1n1_vaccine (Binary: 0 = Not Vaccinated, 1 = Vaccinated)

### Data Definition:

- **unique_id**: Unique identifier for each respondent - (Numerical)    

- **h1n1_worry**: Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried - (Categorical)

- **h1n1_awareness**: Signifies the amount of knowledge or understanding the respondent has about h1n1 flu - (0,1,2) - 0=No knowledge, 1=little knowledge, 2=good knowledge- (Categorical) 
 
- **antiviral_medication**: Has the respondent taken antiviral vaccination - (0,1) (Categorical)
    
- **contact_avoidance**: Has avoided any close contact with people who have flu-like symptoms  - (0,1) - (Categorical)
    
- **bought_face_mask**: Has the respondent bought mask or not - (0,1) - (Categorical)
    
- **wash_hands_frequently**: Washes hands frequently or uses hand sanitizer - (0,1) - (Categorical)
    
- **avoid_large_gatherings**: Has the respondent reduced time spent at large gatherings - (0,1) - (Categorical)
    
- **reduced_outside_home_cont**: Has the respondent reduced contact with people outside own house - (0,1) - (Categorical)
    
- **avoid_touch_face**: Avoids touching nose, eyes, mouth - (0,1) - (Categorical)

- **dr_recc_h1n1_vacc**: Doctor has recommended h1n1 vaccine - (0,1) - (Categorical)
    
- **dr_recc_seasonal_vacc**: Doctor has recommended seasonalflu vaccine - (0,1) - (Categorical)
    
- **chronic_medic_condition**: Has any chronic medical condition - (0,1) - (Categorical)
    
- **cont_child_undr_6_mnth** - Has a regular contact with child the age of 6 months - (0,1) - (Categorical)

- **is_health_worker**: Is respondent a health worker - (0,1) - (Categorical)
    
- **has_health_insur**: Does respondent have health insurance - (0,1) - (Categorical)
    
- **is_h1n1_vacc_effective**:  Does respondent think that the h1n1 vaccine is effective - (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective) - (Categorical)

- **is_h1n1_risky**: What respondenst think about the risk of getting ill with h1n1 in the absence of the vaccine- (1,2,3,4,5)- (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky) - (Categorical)
 
- **sick_from_h1n1_vacc**: Does respondent worry about getting sick by taking the h1n1 vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5Respondent is very worried) - (Categorical)

- **is_seas_vacc_effective**: Does respondent think that the seasonal vaccine is effective- (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective) - (Categorical)

- **is_seas_flu_risky**: What respondenst think about the risk of getting ill with seasonal flu in the absence of the vaccine- (1,2,3,4,5)- (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky) - (Categorical)
 
- **sick_from_seas_vacc**: Does respondent worry about getting sick by taking the seasonal flu vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5Respondent is very worried) - (Categorical)

- **age_bracket** - Age bracket of the respondent - (18 - 34 Years, 35 - 44 Years, 45 - 54 Years, 55 - 64 Years, 64+ Years) - (Categorical)
    
- **qualification** - Qualification/education level of the respondent as per their response -(<12 Years, 12 Years, College Graduate, Some College) - (Categorical)
    
- **race**: Respondent's race - (White, Black, Other or Multiple ,Hispanic) - (Categorical) 
    
- **sex**: Respondent's sex - (Female, Male) - (Categorical)
    
- **income_level**:Annual income of the respondent as per the 2008 poverty Census - (<=$75000-Above Poverty, >$75000, Below Poverty) - (Categorical)
    
- **marital_status**: Respondent's marital status - (Not Married, Married) - (Categorical)
    
- **housing_status**: Respondent's housing status - (Own, Rent) - (Categorical)
    
- **employment**: Respondent's employment status - (Not in Labor Force, Employed, Unemployed) - (Categorical)
    
- **census_msa**: Residence of the respondent with the MSA(metropolitan statistical area)(Non-MSA, MSA-Not Principle, CityMSA-Principle city) - (Yes, no) - (Categorical)
    
- **no_of_adults**:  Number of adults in the respondent's house (0,1,2,3) - (Yes, no) - (Categorical)

- **no_of_children**: Number of children in the respondent's house(0,1,2,3) - (Yes, No) - (Categorical)

- **h1n1_vaccine**: (Dependent variable)Did the respondent received the h1n1 vaccine or not(1,0) - (Yes, No) - (Categorical)

## Project Workflow:
1. **Data Collection & Preprocessing**:
   - Handling missing values
   - Encoding categorical variables
   - Splitting data into training and testing sets
2. **Exploratory Data Analysis (EDA)**:
   - Understanding patterns and trends in vaccine hesitancy
   - Visualizing feature distributions
3. **Model Selection and Training**:
   - Implementing Logistic Regression
   - Evaluating model performance using accuracy, precision, recall, and F1-score
4. **Prediction and Insights**:
   - Interpreting model outputs
   - Identifying key factors affecting vaccine adoption

## Table of Contents:
1. Introduction
2. Project Overview
3. Problem Statement
4. Dataset Information
5. Data Definition
6. Project Workflow
7. Results
8. Future Enhancements
9. Conclusion

## Results:
- The **Logistic Regression model** achieved **high accuracy** in predicting vaccine usage.
- Identified key features influencing vaccine adoption, such as **age, education, and health conditions**.
- Visualized insights into vaccine hesitancy trends among different demographics.

## Future Enhancements:
- Implement **advanced machine learning models** like Random Forest or Neural Networks for better accuracy.
- Incorporate **real-time data** to track changing trends in vaccine hesitancy.
- Develop an interactive **dashboard** for policymakers to analyze vaccination trends dynamically.
- Include **social media sentiment analysis** to capture real-time public perception regarding vaccines.

## Conclusion:
This project successfully predicts vaccine usage using the **Logistic Regression Algorithm**. The insights derived from the model help identify key factors influencing vaccination decisions. The results can assist policymakers in designing effective awareness campaigns and targeted interventions to improve vaccination rates.


## 📧  Feedback and Suggestions:

Thank you for visiting my repository! If you have any questions or feedback, feel free to reach out.

I’d love to hear your thoughts, feedback, and suggestions! Feel free to connect with me:

 LinkedIn: [Shivanand Nashi](https://www.linkedin.com/in/shivanand-s-nashi-79579821a)
 
 Email: shivanandnashi97@gmail.com


Looking forward to connecting and exchanging ideas!

## ✨ Support this project!
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!
Your support helps keep the project active and encourages further development.

Thank you for your support! 💖












