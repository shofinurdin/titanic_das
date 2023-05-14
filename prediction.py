import streamlit as st
import joblib
import pandas as pd
import numpy as np



def load_model():
	model_gb=joblib.load('model.joblib')
	return model_gb

def run_predict_app():
    st.write("Prediction Section")
    model=load_model()
    with st.sidebar:
        st.title("Features")
    # st.sidebar
        pclass= st.selectbox('PClass :',(1,2,3))
        sex = st.selectbox('Sex :', (1,2))
        single= st.selectbox('Single :', (0,1))
        parch= st.selectbox('Parch :', (0,1,2,3,4))
     
    if st.button("Click Here to Classify"):
        dfvalues = pd.DataFrame(list(zip([pclass],[sex],[single],[parch])),columns =['pclass', 'sex', 'single', 'parch'])
        input_variables = np.array(dfvalues[['pclass', 'sex', 'single', 'parch']])
        st.write('Input :')
        data_input=pd.DataFrame(data=input_variables, columns=['pclass', 'sex', 'single', 'parch'])
        st.dataframe(data_input)
        prediction = model.predict(input_variables)
        st.write('Prediction :')
        st.write(prediction)
        if prediction == 1:
            st.image('https://img.freepik.com/premium-vector/emoji-pleasantly-surprised-emoticon-reaction-icon-vector_81894-5337.jpg?w=2000')
        else:
            st.image('https://img.freepik.com/premium-vector/sad-emoticon-yellow-apps-websites_340607-156.jpg?w=2000')
        