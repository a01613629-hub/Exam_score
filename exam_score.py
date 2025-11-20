import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificacion ''')
st.image("1234.webp", caption="Predicción de calificacion.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Horas estudiadas = st.number_input('Horas estudiadas:', min_value=0, max_value=100, value = 0)
  horas sueño = st.number_input('horas sueño:',  min_value=0, max_value=24, value = 0)
  asistencia = st.number_input('asistencia:', min_value=0, max_value=100, value = 0)
  calif. pasadas = st.number_input('calif. pasadas:', min_value=0, max_value=100, value = 0)
  calificacion = st.number_input('calificacion:', min_value=0, max_value=100, value = 0)
  


  user_input_data = {'hours_studied': Horas estudiadas,
                     'sleep_hours': horas sueño,
                     'attendance_percent': asistencia,
                     'previous_scores': calif. pasadas,
                     'exam_score': calificacion
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('exam_score', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613629)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores'] 

st.subheader('Cálculo de calificacion')
st.write('La calificacion del estudiante es ', prediccion)
