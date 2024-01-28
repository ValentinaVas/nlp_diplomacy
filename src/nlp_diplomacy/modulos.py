# -*- coding: utf-8 -*-
from orquestador2.step import Step

#import random as rd
#from datetime import datetime , timedelta
#from pytz import timezone


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification, GPT2Tokenizer
from sklearn.metrics import accuracy_score, classification_report


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string


def separar_mensajes(df):
    mensajes_separados = {'messages': [], 'sender_labels': []}

    for indice, fila in df.iterrows():
        mensajes = fila['messages']
        etiquetas = fila['sender_labels']

        for mensaje, etiqueta in zip(mensajes, etiquetas):
            nuevo_registro = {'messages': mensaje, 'sender_labels': etiqueta}
            mensajes_separados['messages'].append(mensaje)
            mensajes_separados['sender_labels'].append(etiqueta)

    nuevo_df = pd.DataFrame(mensajes_separados)
    return nuevo_df

def unir_jsonl(archivo1, archivo2, archivo_resultado):
    # Leer datos del primer archivo JSONL
    with open(archivo1, 'r') as f1:
        datos1 = f1.readlines()

    # Leer datos del segundo archivo JSONL
    with open(archivo2, 'r') as f2:
        datos2 = f2.readlines()

    # Unir los datos
    datos_resultado = datos1 + datos2

    # Escribir el resultado en un nuevo archivo JSONL
    with open(archivo_resultado, 'w') as f_resultado:
        f_resultado.writelines(datos_resultado)

def plot_cm(cm):
    classes=['Mentira','Verdad']
    df_cm=pd.DataFrame(cm, index=classes, columns=classes)
    ax=sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Objetivo")

class exploratory_analysis(Step):
    def ejecutar(self):
        # Ruta datos
        ruta_data=self.getFolderPath() + "data/"

        # unir datos de prueba y entrenamiento
        archivo1 = ruta_data + 'test.jsonl'
        archivo2 = ruta_data + 'train.jsonl'
        archivo_resultado =ruta_data + 'data.jsonl'

        # Llama a la función para unir los archivos
        unir_jsonl(archivo1, archivo2, archivo_resultado)

        # Pasar datos dataFrame
        data = pd.read_json(archivo_resultado, orient='records', lines=True)

        # Pasar datos dataFrame test
        data_test = pd.read_json(archivo_resultado, orient='records', lines=True)
        # Pasar datos dataFrame train
        data_train = pd.read_json(archivo_resultado, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_1=separar_mensajes(data)
        # Eliminar duplicados
        data_2 = data_1.drop_duplicates()
        # Separar mensajes y etiquetas de sender_labels
        data_test=separar_mensajes(data_test)
        # Separar mensajes y etiquetas de sender_labels
        data_train=separar_mensajes(data_train)

        conteo_etiquetas = data_2['sender_labels'].apply(pd.Series).stack().value_counts()

        print(conteo_etiquetas)
        # Crea un histograma
        conteo_etiquetas.plot(kind='bar', color=['green', 'red'])
        plt.title('Histograma de Etiquetas')
        plt.xlabel('Etiqueta')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0)
        plt.show()

        # Guardar histograma
        # ruta para analisis exploratorio
        ruta_analyze = self.getFolderPath() + "analyze/"

        # Guarda el histograma
        plt.savefig(ruta_analyze + 'hist_sender_labels.png')

        # Descarga de recursos adicionales de nltk
        nltk.download('punkt')
        nltk.download('stopwords')

        # Concatenar todos los mensajes en un solo texto
        all_messages = ' '.join(data_2['messages'])

        # Tokenizar las palabras
        tokens = word_tokenize(all_messages)

        # Eliminar stopwords y signos de puntuación
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

        # Calcular la frecuencia de las palabras
        freq_dist = FreqDist(filtered_tokens)

        # Mostrar las 20 palabras más frecuentes y su frecuencia
        print(freq_dist.most_common(20))

        # Visualizar un histograma
        plt.figure(figsize=(12, 6))
        freq_dist.plot(20, cumulative=False)
        plt.title('Histograma de Frecuencia de Palabras')
        plt.xlabel('Palabra')
        plt.ylabel('Frecuencia')
        plt.show()

        # Guarda el histograma
        plt.savefig(ruta_analyze + 'hist_frequency.png')


class model_LogisticRegression(Step):
    def ejecutar(self):

        # Ruta datos
        ruta_data=self.getFolderPath() + "data/"

        # ruta datos
        archivo_resultado =ruta_data + 'data.jsonl'
        # ruta test
        archivo_test=ruta_data + 'test.jsonl'
        # ruta traint
        archivo_train=ruta_data + 'train.jsonl'
  
        # Pasar datos dataFrame
        data = pd.read_json(archivo_resultado, orient='records', lines=True)
        # Pasar datos dataFrame test
        data_test = pd.read_json(archivo_resultado, orient='records', lines=True)
        # Pasar datos dataFrame train
        data_train = pd.read_json(archivo_resultado, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_1=separar_mensajes(data)
        # Separar mensajes y etiquetas de sender_labels
        data_test=separar_mensajes(data_test)
        # Separar mensajes y etiquetas de sender_labels
        data_train=separar_mensajes(data_train)

#        conteo_etiquetas = data_1['sender_labels'].apply(pd.Series).stack().value_counts()
#
#        print(conteo_etiquetas)
#        # Crea un histograma
#        conteo_etiquetas.plot(kind='bar', color=['green', 'red'])
#        plt.title('Histograma de Etiquetas')
#        plt.xlabel('Etiqueta')
#        plt.ylabel('Frecuencia')
#        plt.xticks(rotation=0)
#        plt.show()
#
#        # Guardar histograma
#        # ruta para analisis exploratorio
#        ruta_analyze = self.getFolderPath() + "analyze/"
#
#        # Guarda el histograma
#        plt.savefig(ruta_analyze + 'hist_sender_labels.png')

        # Nueva columna con 1 si es verdad y 0 si es mentira
        target_map={True:1,False:0}
        data_1['target']=data_1['sender_labels'].map(target_map)

        # Separar los datos de entrenamiento y prueba
        df_train, df_test = train_test_split(data_1)

        print(df_train)
        print(df_test)
        
        #
        vectorizer= TfidfVectorizer(max_features=2000)

        # Vectorizar datos de prueba y entrenamiento
        x_train=vectorizer.fit_transform(df_train['messages'])
        x_test=vectorizer.fit_transform(df_test['messages'])
        
        y_train= df_train['target']
        y_test= df_test['target']

        model= LogisticRegression(max_iter=1000)
        model.fit(x_train,y_train)
        print("Train acc:", model.score(x_train,y_train))
        print("Test acc:", model.score(x_test,y_test))

        p_train= model.predict(x_train)
        p_test=model.predict(x_test)
        
        #Matriz de confusion train
        cm_train= confusion_matrix(y_train,p_train,normalize='true')
        print(cm_train)

        #Matriz de confusion test
        cm_test= confusion_matrix(y_test,p_test,normalize='true')
        print(cm_test)

#        def plot_cm(cm):
#            classes=['Mentira','Verdad']
#            df_cm=pd.DataFrame(cm, index=classes, columns=classes)
#            ax=sn.heatmap(df_cm, annot=True, fmt='g')
#            ax.set_xlabel("Prediccion")
#            ax.set_ylabel("Objetivo")
        
        # Mostrar matriz confusión para train
        plot_cm(cm_train)
        plt.show()

        # Mostrar matriz confusión para test
        plot_cm(cm_test)
        plt.show()

        # visualizar word index map
        word_index_map= vectorizer.vocabulary_
        print(word_index_map)

        #Determinar el peso de las palabras asociadas a la verdad
        model.coef_[0]
        corte=4
        print("Palabras asociadas a la vedad")
        for word, index in word_index_map.items():
            weight= model.coef_[0][index]
            if weight > corte:
                print(word,weight)

        #Determinar el peso de las palabras asociadas a la mentira
        model.coef_[0]
        corte=4
        print("Palabras asociadas a la vedad")
        for word, index in word_index_map.items():
            weight= model.coef_[0][index]
            if weight < corte:
                print(word,weight)
        
        # Histograma peso de las palabras
        plt.hist(model.coef_[0], bins=30)
        plt.show

        # Validación del modelo con los datos validation.jsonl REVISAR SI ES NECESARIO REPLICART EL CODIGO ANTERIOR CON MATRIZ DE CONFUSION
        # Leer datos  validation
        archivo_validation=ruta_data + 'validation.jsonl'
  
        # Pasar datos dataFrame
        data_validation = pd.read_json(archivo_validation, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_validation=separar_mensajes(data)

         # Vectorizar datos de prueba y entrenamiento
        x_validation=vectorizer.fit_transform(data_validation['messages'])

        data_validation["predict"]=model.predict(x_validation)
        
        
class model_LSTM(Step):
    def ejecutar(self):
        
        # Ruta datos
        ruta_data=self.getFolderPath() + "data/"

        # Leer datos
        archivo_resultado =ruta_data + 'data.jsonl'

        # Pasar datos dataFrame
        data = pd.read_json(archivo_resultado, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_1=separar_mensajes(data)

        # Preprocesamiento de datos
        encoder = LabelEncoder()
        data_1['sender_labels'] = encoder.fit_transform(data_1['sender_labels'])

        # Dividir los datos en conjuntos de entrenamiento y prueba
        train_data, test_data = train_test_split(data_1, test_size=0.2, random_state=42)

        # Tokenizar y secuenciar los mensajes
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_data['messages'])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['messages']), maxlen=50)
        X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['messages']), maxlen=50)

        # Crear el modelo de aprendizaje profundo
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=50))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))

        # Compilar el modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo
        y_train = train_data['sender_labels']
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

        # Evaluar el modelo
        y_test = test_data['sender_labels']
        accuracy = model.evaluate(X_test, y_test)
        print(f'Accuracy: {accuracy}')

        # Mostrar la matriz de confusión
        conf_matrix = confusion_matrix(X_test, y_test)
        print('Confusion Matrix:')
        print(conf_matrix)

        # Calcular y mostrar sensibilidad y especificidad
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')

        # Mostrar el informe de clasificación
        print('Classification Report:')
        print(classification_report(X_test, y_test))

        # Hacer predicciones
        predictions = model.predict(X_test)
        predictions = np.round(predictions).astype(int)
        test_data['predicted_labels'] = encoder.inverse_transform(predictions)

        #---------------------------------- quitar lo que esta repetido
        # Hacer predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)

        # Convertir las etiquetas a formato binario (verdadero/falso)
        y_true = test_data['sender_labels'].values

        # Calcular la precisión del modelo
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')

        # Mostrar la matriz de confusión
        conf_matrix = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(conf_matrix)

        # Calcular y mostrar sensibilidad y especificidad
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')

        # Mostrar el informe de clasificación
        print('Classification Report:')
        print(classification_report(y_true, y_pred))


    

