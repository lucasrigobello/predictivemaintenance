# Case Predictive Maintenance - API com FastAPI
[![Docker Image CI](https://github.com/lucasrigobello/predictivemaintenance/actions/workflows/docker-image.yml/badge.svg)](https://github.com/lucasrigobello/predictivemaintenance/actions/workflows/docker-image.yml)

## 📌 Sobre o projeto
Este projeto é um case fictício de manutenção preditiva inspirado em cenários reais da indústria de celulose e papel. O objetivo é desenvolver modelos de Machine Learning capazes de prever falhas em ativos críticos, reduzindo custos de manutenção e mantendo a disponibilidade operacional.
- Classificação Close to Failure: prever se um ativo está a menos de 20 ciclos da falha.
- Predição de Remaining Useful Life (RUL): estimar o tempo restante até a falha de um ativo.
Os modelos são expostos via FastAPI, permitindo integração com sistemas de manutenção e monitoramento.

## 🚀 Tecnologias utilizadas
- **Python** (para implementação do modelo e da API)
- **FastAPI** (para exposição do modelo via API REST)
- **Scikit-learn** ( para framework para machine learning)
- **Docker** (para conteinerização da aplicação)
- **Kubernetes** (para orquestração e deploy)
- **Swagger** (para documentação da API)

## 📂 Estrutura do projeto
```bash
├── src
│   ├── classes/              # Código de objetos
│   ├── config/               # Configurações gerais do projeto
│   ├── models/               # Código relacionado ao treinamento do modelo
│   ├── utils/                # Código de funções do modelo
│   └── main.py               # Ponto de entrada da API
├── data/
│   ├── train.txt             # Dataset de treino
│   └── test.txt              # Dataset de teste
│
├── eda notebook/             # Notebook com resultados de EDA e Treinamento do modelo
├── .github/                  # Workflows para Github Action
├── helm-charts/              # Manifests para deploy no Kubernetes
├── Dockerfile                # Configuração do container Docker
├── requirements.txt          # Dependências do projeto
├── LICENSE                   # Licença MIT
├── README.md                 # Documentação do projeto
└── .gitignore                # Arquivos ignorados no repositório Git
````

## 🛠 Como configurar o projeto
1.	Clone este repositório:

```bash
git clone https://github.com/lucasrigobello/predictivemaintenance.git
cd predictivemaintenance
````

2.	Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

3.  Para iniciar a API:
```bash
python src/main.py 
```

## 🖥️ Utilização da API
A API expõe um endpoint para prever a Close to Failyre (Classificação) e predição por regressão de RUL:

- **GET** ```/classificar/{id}``` 
    - **Parâmetros:** ```id``` (ID do Asset a ser classificado)
    - **Retorno:** ```{"0": "Probabilidade de Close to Failure"}```

- **GET** ```/predict/{id}``` 
    - **Parâmetros:** ```id``` (ID do Asset a ser avaliado RUL)
    - **Retorno:** ```{"0": "RUL"}```


Exemplo de requisição:
```bash
curl -X GET "http://localhost:8000/classificar/8"
```

## 📦 Executando com Docker
Para construir e executar o container:
```bash
docker build -t predictivemaintenance .
docker run -p 8000:8000 predictivemaintenance
```

## ☁️ Deploy no Kubernetes
Para implantar no Kubernetes, use os manifests disponíveis na pasta ```kubernetes/```:
```bash
kubectl apply -f ./helm-charts/templates/deployment.yaml
kubectl apply -f ./helm-charts/templates/service.yaml
```

## 📖 Documentação
A documentação da API pode ser acessada via Swagger em:
```
http://localhost:8000/docs
```

## 📜 Licença
Este projeto está sob a licença MIT.
________________________________________