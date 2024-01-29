# -*- coding: utf-8 -*-
from orquestador2.step import Step

import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
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


def separate_messages(df):
    df_new = {'messages': [], 'sender_labels': [],'receiver_labels':[]}

    for indice, fila in df.iterrows():
        menssages = fila['messages']
        sender_labels = fila['sender_labels']
        receiver_labels=fila['sender_labels']

        for menssage, sender_label,receiver_label in zip(menssages, sender_labels,receiver_labels):
            #nuevo_registro = {'messages': menssage, 'sender_labels':sender_label,'receiver_label':receiver_label}
            df_new['messages'].append(menssage)
            df_new['sender_labels'].append(sender_label)
            df_new['receiver_labels'].append(receiver_label)

    df_result = pd.DataFrame(df_new)
    return df_result

def unir_jsonl(file1, file2,file3, file_result):
    # Leer datos del primer file JSONL
    with open(file1, 'r') as f1:
        data1 = f1.readlines()

    # Leer datos del segundo file JSONL
    with open(file2, 'r') as f2:
        data2 = f2.readlines()
    
    with open(file3, 'r') as f2:
        data3 = f2.readlines()

    # Unir los datos
    data_result = data1 + data2 + data3

    # Escribir el resultado en un nuevo file JSONL
    with open(file_result, 'w') as f_result:
        f_result.writelines(data_result)

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
        file1 = ruta_data + 'test.jsonl'
        file2 = ruta_data + 'train.jsonl'
        file3 = ruta_data + 'validation.jsonl'
        file_result =ruta_data + 'data.jsonl'

        # Llama a la función para unir los files
        unir_jsonl(file1, file2,file3, file_result)

        # Pasar datos dataFrame
        data = pd.read_json(file_result, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_1=separate_messages(data)
        # Eliminar duplicados
        data_2 = data_1.drop_duplicates()
    
        conteo_etiquetas = data_2['sender_labels'].apply(pd.Series).stack().value_counts()

        print(conteo_etiquetas)
        # Crea un histograma
        conteo_etiquetas.plot(kind='bar', color=['green', 'red'])
        plt.title('Histograma de Etiquetas')
        plt.xlabel('Etiqueta')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0)

        # Guardar histograma
        # ruta para analisis exploratorio
        ruta_analyze = self.getFolderPath() + "analyze/"
        # Guarda el histograma
        plt.savefig(ruta_analyze + 'hist_sender_labels.png')
        plt.show()

        # Descarga de recursos adicionales de nltk
        nltk.download('punkt')
        nltk.download('stopwords')

        # Concatenar todos los mensajes en un solo texto
        all_messages = ' '.join(data_2['messages'])

        # Pasar todo a minisculas
        all_messages=all_messages.lower()

        # Tokenizar las palabras
        tokens = word_tokenize(all_messages)

        # Eliminar stopwords y signos de puntuación
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

        # Calcular la frecuencia de las palabras
        freq_dist = FreqDist(filtered_tokens)

        # Calcular la cantidad total de palabras
        total_words = len(filtered_tokens)

        # Imprimir la cantidad total de palabras
        print(f'Cantidad total de palabras: {total_words}')

        # Mostrar las 100 palabras más frecuentes y su frecuencia
        print(freq_dist.most_common(100))

        # Visualizar el histograma
        plt.figure(figsize=(12, 6))
        freq_dist.plot(50, cumulative=False)
        plt.title('Histograma de Frecuencia de Palabras')
        plt.xlabel('Palabra')
        plt.ylabel('Frecuencia')
        
        # Guarda el histograma
        plt.savefig(ruta_analyze + 'hist_frequency.png')
        plt.close()
        

class model_LogisticRegression(Step):
    def ejecutar(self):

        # Ruta datos
        ruta_data=self.getFolderPath() + "data/"

        # ruta datos
        file_resultado =ruta_data + 'data.jsonl'

        # Pasar datos dataFrame
        data = pd.read_json(file_resultado, orient='records', lines=True)
    
        # Separar mensajes y etiquetas de sender_labels
        data_1=separate_messages(data)
        
        # Nueva columna con 1 si es verdad y 0 si es mentira
        target_map={True:1,False:0}
        data_1['target']=data_1['sender_labels'].map(target_map)

        # Separar los datos de entrenamiento y prueba
        df_train, df_test = train_test_split(data_1)

        print(df_train)
        print(df_test)
        
        #vectorizar
        vectorizer= TfidfVectorizer(max_features=2000)

        # Vectorizar datos de prueba y entrenamiento
        x_train=vectorizer.fit_transform(df_train['messages'])
        x_test=vectorizer.fit_transform(df_test['messages'])
        
        # Capturar las etiquetas
        y_train= df_train['target']
        y_test= df_test['target']

        # Modelo de regresión logistica con 1000 iteraciones
        model= LogisticRegression(max_iter=1000)
        model.fit(x_train,y_train)

        #Validar modelo
        print("Train acc:", model.score(x_train,y_train))
        print("Test acc:", model.score(x_test,y_test))

        # Hacer predicciones para train y test
        p_train= model.predict(x_train)
        p_test=model.predict(x_test)
        
        #Matriz de confusion train
        cm_train= confusion_matrix(y_train,p_train,normalize='true')
        print(cm_train)

        #Matriz de confusion test
        cm_test= confusion_matrix(y_test,p_test,normalize='true')
        print(cm_test)
     
        # Mostrar matriz confusión para train
        plot_cm(cm_train)
        plt.show()

        # Mostrar matriz confusión para test
        plot_cm(cm_test)
        plt.show()

        # word index map
        word_index_map= vectorizer.vocabulary_
        #print(word_index_map)

        #Determinar el peso de las palabras asociadas a la verdad
        model.coef_[0]
        
        # Histograma peso de las palabras
        plt.hist(model.coef_[0], bins=30)
        plt.show()

        
        
class model_LSTM(Step):
    def ejecutar(self):
        
        # Ruta datos
        ruta_data=self.getFolderPath() + "data/"

        # Leer datos
        file_result =ruta_data + 'data.jsonl'

        # Pasar datos dataFrame
        data = pd.read_json(file_result, orient='records', lines=True)

        # Separar mensajes y etiquetas de sender_labels
        data_1=separate_messages(data)

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
      
        # Hacer predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)

        # Capturar sender_labels
        y_true = test_data['sender_labels'].values

        # Mostrar la matriz de confusión
        conf_matrix = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(conf_matrix)

        # Mostrar matriz confusión para test
        plot_cm(conf_matrix)
        plt.show()

        # Calcular y mostrar sensibilidad y especificidad
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

        print(f'Sensitivity: {sensitivity}')
        print(f'Specificity: {specificity}')

        # Mostrar el informe de clasificación
        print('Classification Report:')
        print(classification_report(y_true, y_pred))


