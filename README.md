#  Projeto de Monitoramento de Qualidade Ambiental (Padrão Senior Data Scientist)

Este projeto é um sistema completo de Machine Learning para classificação de qualidade ambiental, estruturado como um pipeline automatizado seguindo as práticas de MLOps e o modelo CRISP-DM. Ele foi refatorado para seguir padrões de engenharia de dados e ciência de dados de nível sênior.

##  Tecnologias Utilizadas
- **Python** (Linguagem principal)
- **Optuna** (Otimização de hiperparâmetros com busca expandida)
- **LazyPredict** (Benchmark inicial de modelos com ranking visual)
- **MLflow** (Rastreamento de experimentos com backend SQLite)
- **Streamlit** (Interface web com dashboard e inferência)
- **Docker** (Containerização completa)
- **ydata-profiling** (EDA automática expandida com gráficos customizados)
- **Scikit-learn** (Pipeline, Validação Cruzada Estratificada e Modelos)
- **Imbalanced-learn** (SMOTE k=3 para balanceamento robusto)

##  Estrutura do Projeto
```
├── data/               # Dataset bruto
├── models/             # Modelos, transformadores e feature_names salvos
├── notebooks/          # Notebooks de experimentação interativa
├── src/                # Código fonte modularizado
│   ├── eda.py          # EDA completa, outliers e gráficos PNG
│   ├── preprocessing.py# Limpeza, Imputer, SMOTE e Stats
│   └── training.py     # Optuna, MLflow, Métricas Macro e Validação
├── app/                # Aplicação Streamlit (Dashboard + Previsão)
├── reports/            # Relatórios HTML e pasta figures/ com PNGs
├── main.py             # Orquestrador do pipeline completo
├── config.yaml         # Configurações centralizadas
├── requirements.txt    # Dependências do projeto
└── Dockerfile          # Configuração do container para deploy
```

##  Como Executar

### 1. Instalação Local
```bash
pip install -r requirements.txt
```

### 2. Executar o Pipeline Completo
O `main.py` orquestra todas as etapas: EDA -> Pré-processamento -> Benchmark -> Otimização -> Treinamento Final -> Verificação de Artefatos.
```bash
python main.py
```

### 3. Visualizar Experimentos (MLflow)
Para ver o histórico de execuções, métricas macro e artefatos (matriz de confusão, importância de features):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 4. Executar a Aplicação Web
```bash
streamlit run app/app.py
```

### 5. Docker
```bash
docker build -t ambiental-ml .
docker run -p 8501:8501 ambiental-ml
```

##  Diferenciais Implementados
- **Limpeza Inteligente:** Remoção automática de colunas constantes, alta cardinalidade e redundantes (correlação > 0.95).
- **No Data Leakage:** Separação rigorosa entre treino e teste; Imputer e Scaler fitados apenas no treino.
- **Métricas Macro:** Foco no desempenho real em todas as classes, não apenas na majoritária.
- **Verificação de Artefatos:** Teste automático de integridade do modelo salvo antes de finalizar o pipeline.
- **Visualização Completa:** Mais de 10 gráficos gerados automaticamente para análise profunda.

---
*Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.*
