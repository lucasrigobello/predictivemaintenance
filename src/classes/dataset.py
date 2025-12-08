import pandas as pd
import yaml

# Read Features Config YAML file
with open("./src/config/conf.yaml", 'r') as stream:
    conf_features = yaml.safe_load(stream)

train_path = './data/PM_train.txt'
test_path = './data/PM_test.txt'

target_prediction = 'RUL'
target_classification = 'close_to_failure'

class Test_Data:
  __test__ = False
  def __init__(self, id):
    # id == Asset id
    self.id = id

    # load base de dados
    self.df = pd.read_csv(test_path, sep=' ', header=None)
    self.df.dropna(axis=1, how='all', inplace=True)

    # Renomear os dados
    for nameColumn in conf_features['data']['columns_names'].keys():
        first_columns = conf_features['data']['columns_names'][nameColumn]['first']
        last_columns = conf_features['data']['columns_names'][nameColumn]['last']
        for idx, nColumn in enumerate(range(first_columns, last_columns+1)):
            if first_columns == last_columns:
                self.df.rename(columns={nColumn: nameColumn}, inplace=True)
            else:
                self.df.rename(columns={nColumn: f'{nameColumn}{idx+1}'}, inplace=True)

    self.df_columns = list(self.df.columns)

    # Criando Features - criando as freatures de métricas
    window_size = 3
    for coluna in self.df.columns.drop(['Asset id', 'runtime']):
        self.df[f'{coluna}-mean'] = ''
        self.df[f'{coluna}-min'] = ''
        self.df[f'{coluna}-max'] = ''
        self.df[f'{coluna}-std'] = ''

        for asset in self.df['Asset id'].unique():
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-mean'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).mean()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-min'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).min()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-max'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).max()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-std'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).std()

  def X(self):
    X_test_test = pd.DataFrame([], columns=self.df.columns)
    for asset in self.df['Asset id'].unique():
        X_test_test = pd.concat([X_test_test, 
                                pd.DataFrame(self.df[self.df['Asset id']==asset].iloc[-1]).T])
    
    self.item = X_test_test[X_test_test['Asset id'] == int(self.id)]

    # return X_test_test.drop(columns=['Asset id'])
    return self.item.drop(columns=['Asset id'])
  
class Train_Data:
  def __init__(self):
    # load base de dados
    self.df = pd.read_csv(train_path, sep=' ', header=None)
    self.df.dropna(axis=1, how='all', inplace=True)

    # Renomear os dados
    for nameColumn in conf_features['data']['columns_names'].keys():
        first_columns = conf_features['data']['columns_names'][nameColumn]['first']
        last_columns = conf_features['data']['columns_names'][nameColumn]['last']
        for idx, nColumn in enumerate(range(first_columns, last_columns+1)):
            if first_columns == last_columns:
                self.df.rename(columns={nColumn: nameColumn}, inplace=True)
            else:
                self.df.rename(columns={nColumn: f'{nameColumn}{idx+1}'}, inplace=True)

    self.df_columns = list(self.df.columns)

    # Criando Features - criando as freatures de métricas
    window_size = 3
    for coluna in self.df.columns.drop(['Asset id', 'runtime']):
        self.df[f'{coluna}-mean'] = ''
        self.df[f'{coluna}-min'] = ''
        self.df[f'{coluna}-max'] = ''
        self.df[f'{coluna}-std'] = ''

        for asset in self.df['Asset id'].unique():
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-mean'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).mean()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-min'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).min()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-max'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).max()
            self.df.loc[self.df['Asset id'] == asset, f'{coluna}-std'] = self.df.loc[self.df['Asset id'] == asset, coluna].rolling(window_size).std()

    # Criação de features
    ## Remaining Useful Life (RUL)
    self.df['RUL'] = [ self.df[self.df['Asset id']==dataRow['Asset id']]['runtime'].max() - dataRow['runtime'] for idx, dataRow in self.df.iterrows() ]

    ## Classificação Close to Failure
    self.df['close_to_failure'] = False
    self.df.loc[self.df['RUL']<20,'close_to_failure'] = True

    ## Composição do DataFrame
    self.df.drop(columns='Asset id', inplace=True)
    self.df.dropna(inplace=True)

  def X(self):
    return self.df.drop(columns=[target_classification, target_prediction])
  
  def Y_classification(self):
    return self.df[target_classification]
  
  def Y_predition(self):
    return self.df[target_prediction]