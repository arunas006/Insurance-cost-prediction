import streamlit as st
import pickle
import sklearn
import numpy as np

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>**Insurance Preimum Prediction App **</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    Age_Bucket=st.selectbox("Select The Correct Age Bucket",["less 30 Yrs","31-40 Yrs","41-50 Yrs","51-55 Yrs","greater 55 Yrs"])

with col2:
    Diabetes=st.selectbox("Do you have Diabetics Issue",["Yes","No"])

with col3:
    BloodPressureProblems=st.selectbox("Do you have Blood Pressure Issue",["Yes","No"])

col4, col5, col6 = st.columns(3)

with col4:
    AnyTransplants=st.selectbox("Any organ transplant done?",["Yes","No"])

with col5:
    AnyChronicDiseases=st.selectbox("Do you affected by any Chronic Diseases Issue",["Yes","No"])

with col6:
    KnownAllergies=st.selectbox("Do you have any Alergic issues",["Yes","No"])

col7, col8= st.columns(2)

with col7:
    Weight=st.slider("Enter Your Weight",min_value=40,max_value=150,step=1)

with col8:
    Height=st.slider("Enter Your Heightt",min_value=70,max_value=230,step=1)

col9, col10= st.columns(2)

with col9:
    HistoryOfCancerInFamily=st.selectbox("Do you have any Cancer History in the family",["Yes","No"])  

with col10:
    NumberOfMajorSurgeries=st.selectbox("Have you undergone any major surgery",[0,1,2,3])  

encode_list={'Diabetes':{'Yes':1,'No':0},
             'BloodPressureProblems':{'Yes':1,'No':0},
             'AnyTransplants':{'Yes':1,'No':0},
             'AnyChronicDiseases':{'Yes':1,'No':0},
             'KnownAllergies':{'Yes':1,'No':0},
             'HistoryOfCancerInFamily':{'Yes':1,'No':0},
             'HistoryOfCancerInFamily':{'Yes':1,'No':0},
             'Age_Bucket':{"less 30 Yrs":1,
                           "31-40 Yrs":1,
                           "41-50 Yrs":1,
                           "51-55 Yrs":1,
                           "greater 55 Yrs":1}
             }



def model_prediction(Diabetes,BloodPressureProblems, AnyTransplants,AnyChronicDiseases,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries,Height,Weight,Age_Bucket,Healthy=0,\
                     A=0,B=0,C=0,D=0,E=0):
    BMI=Weight/(Height/100)**2
    Diabetes_info=encode_list['Diabetes'][Diabetes]
    BloodPressureProblems_info=encode_list['BloodPressureProblems'][BloodPressureProblems]
    AnyTransplants_info=encode_list['AnyTransplants'][AnyTransplants]
    AnyChronicDiseases_info=encode_list['AnyChronicDiseases'][AnyChronicDiseases]
    KnownAllergies_info=encode_list['KnownAllergies'][KnownAllergies]
    HistoryOfCancerInFamily_info=encode_list['HistoryOfCancerInFamily'][HistoryOfCancerInFamily]
    

    if (Diabetes_info+BloodPressureProblems_info+AnyTransplants_info+AnyChronicDiseases_info+KnownAllergies_info+HistoryOfCancerInFamily_info+NumberOfMajorSurgeries)==0:
        Healthy=1
    
    if Age_Bucket=='less 30 Yrs':
        A=1
    elif Age_Bucket=='31-40 Yrs':
        B=1
    elif Age_Bucket=='41-50 Yrs':
        C=1
    elif Age_Bucket=='51-55 Yrs':
        D=1
    else:
        E=1

    with open("model.pkl",'rb') as file:
        insurance_predictor=pickle.load(file)

    with open("scaler.pkl",'rb') as f:
        scale=pickle.load(f)
    
    Input_feature=[[Diabetes_info,BloodPressureProblems_info,AnyTransplants_info,AnyChronicDiseases_info,KnownAllergies_info,\
                    HistoryOfCancerInFamily_info,NumberOfMajorSurgeries,Healthy,BMI,B,C,D,A,E]]
    
    scaled_feature=scale.transform(Input_feature)
    
    return np.round(insurance_predictor.predict(scaled_feature),2)[0]


if st.button("PREDICT"):
    output=model_prediction(Diabetes,BloodPressureProblems, AnyTransplants,AnyChronicDiseases,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries,Height,Weight,Age_Bucket,Healthy=0,\
                     A=0,B=0,C=0,D=0,E=0)
    st.write(f"As per the details provided the Insurance Premium will be {output} Rupees")
else:
    st.write("Awaiting for Required features confirmation")
