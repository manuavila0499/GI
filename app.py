import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def cargar_datos():
    dataset = pd.read_csv('bbdd3.csv', encoding='ISO-8859-1', delimiter=';')
    
    # Convertir 'Plazo en dias' y 'Certificacion' a float
    dataset['Plazo en dias'] = dataset['Plazo en dias'].astype(float)
    dataset['Certificacion'] = dataset['Certificacion'].astype(float)
    
    # Preparar X e y
    X = dataset[['Plazo en dias', 'Certificacion', 'Subcontratacion']].copy()
    y = dataset['Coste total'].astype(float)
    
    # Crear y ajustar el LabelEncoder
    labelencoder = LabelEncoder()
    X['Subcontratacion_encoded'] = labelencoder.fit_transform(X['Subcontratacion'])
    
    # Crear y ajustar el ColumnTransformer con OneHotEncoder
    ct = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Plazo en dias', 'Certificacion']),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), ['Subcontratacion_encoded'])
        ]
    )
    
    # Transformar X
    X_transformed = ct.fit_transform(X)
    
    return X_transformed, y, labelencoder, ct, dataset

def entrenar_modelo(X, y):
    modelo = RandomForestRegressor(n_estimators=1050, random_state=42)
    modelo.fit(X, y)
    return modelo

def main():
    st.image('logo.png', width=400)
    st.title("Predicción de Costes")
    
    # Cargar datos y entrenar modelo
    X, y, labelencoder, ct, dataset = cargar_datos()
    modelo = entrenar_modelo(X, y)
    
    st.write("Introduce las variables para predecir el coste:")
    
    # Inputs
    var_3 = st.number_input("Plazo en días", min_value=0.0, value=0.0, step=1.0)
    var_4 = st.number_input("Certificación", min_value=0.0, value=0.0, step=100.0)
    var_8 = st.selectbox("Subcontratación", ["No", "Si"])
    
    if st.button("Predecir"):
        try:
            # Preparar datos de entrada
            var_8_encoded = labelencoder.transform([var_8])[0]
            
            # Crear DataFrame con el formato correcto
            input_data = pd.DataFrame({
                'Plazo en dias': [float(var_3)],
                'Certificacion': [float(var_4)],
                'Subcontratacion_encoded': [var_8_encoded]
            })
            
            # Transformar datos de entrada
            input_transformed = ct.transform(input_data)
            
            # Realizar predicción
            prediccion = modelo.predict(input_transformed)[0]
            
            st.success(f"Coste estimado: {prediccion:.2f} €")
            
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")

if __name__ == "__main__":
    main()