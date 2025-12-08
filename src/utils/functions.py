from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

import joblib
import pandas as pd
# import yaml

from classes.dataset import Train_Data, Test_Data

# #===========================================
# # Read Features Confif YAML file
# with open("./src/config/conf.yaml", 'r') as stream:
#     conf_features = yaml.safe_load(stream)

# target_prediction = 'RUL'
# target_classification = 'close_to_failure'

#===========================================
# Prepara o dado para realizar predict
def preparar_dado_predict(id):
    
    # load base de dados
    X = Test_Data(id).X()
    return X

#===========================================
# Prepara o dado para treinamento - Train e Test Split
def preparar_dado_trainamento_classifictation():
    
    X, Y_classification, Y_predition = get_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_classification, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def preparar_dado_trainamento_prediction():
    
    X, Y_classification, Y_predition = get_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_predition, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

#===========================================
# Gerar relatorio de avaliação do modelo - Retorno em uma tabela HTML
def validation_classification(y_true, y_pred):

    return pd.DataFrame(data=classification_report(y_true, y_pred, output_dict=True)).to_html()

#===========================================
# Realizar o treinmento de novos Modelos
def treinar_modelo_classification(X_train, y_train):
    
    # Criar novo modelo
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42)

    # create a pipeline with preprocessor and classifier
    pipeline = Pipeline([('classifier', clf)
                         ])
    
    modelo = pipeline.fit(X_train, y_train.values.ravel())

    # salvar o Classification Model
    joblib.dump(modelo, "./src/models/pipeline_model_classification.pkl") 

    return modelo

def treinar_modelo_predicction(X_train, y_train):
    
    # Criar novo modelo
    reg = RandomForestRegressor(max_depth=20, random_state=42)

    # create a pipeline with preprocessor and classifier
    pipeline = Pipeline([('regression_model', reg)
                         ])

    modelo = pipeline.fit(X_train, y_train.values.ravel())

    # salvar o Classification Model
    joblib.dump(modelo, "./src/models/pipeline_model_prediction.pkl") 

    return modelo

#===========================================
# Preparar o dado para realizar treinamento
def get_dataset():

    # load base de dados
    X = Train_Data().X()
    Y_classification = Train_Data().Y_classification()
    Y_predition = Train_Data().Y_predition()
    
    return X, Y_classification, Y_predition