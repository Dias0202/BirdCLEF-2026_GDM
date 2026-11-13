# Relatório de Conclusão: Fase 1 - Setup do Ambiente e EDA

A **Fase 1** descrita no plano de implementação foi totalmente concluída e testada. Abaixo detalhamos as ações tomadas e os resultados obtidos.

## 1. Configuração do Ambiente e Dependências
- Criado o arquivo `requirements.txt` contendo as bibliotecas essenciais (`torch`, `torchaudio`, `mlflow`, `pandas`, `librosa`, `soundfile`, `timm`, etc.).
- O arquivo inclui o link para a versão do PyTorch com CUDA (`cu118`/`cu121`), preparando o terreno para a GTX 1650 Super na fase de treinamento.
- **Teste:** Dependências principais foram instaladas no `.venv`.

## 2. Leitura e Processamento de Metadatas (EDA)
- Criado o script `src/eda.py` para automatizar a leitura do dataset a partir do cache do kagglehub (`~/.cache/kagglehub/competitions/birdclef-2026`, configurável por `BIRDCLEF_DATA_DIR`).
- **Teste:** O script leu os arquivos com sucesso:
  - `train.csv`: 35.549 registros.
  - `taxonomy.csv`: 234 classes oficiais.
  - Constatamos empiricamente que as classes em `train.csv` (206) são um subconjunto do `taxonomy.csv` (234), validando a regra do Kaggle (nem todas as espécies têm áudio de treino).

## 3. Dataset e DataLoader Customizado em PyTorch
- Criado o script `src/dataset.py` implementando a classe `BirdCLEFDataset`.
- **Estratégia Adotada:** 
  - Utilização da biblioteca `soundfile` em conjunto com PyTorch para ler eficientemente os metadados de `.ogg` e carregar as amostras sem estourar a RAM.
  - Áudios são cortados aleatoriamente em *chunks* de exatos **5 segundos** (`chunk_length = 5 * 32000`). Se menores que 5s, recebem padding por zeros.
  - Rótulos são convertidos em vetores *Multi-Hot* de tamanho 234.
- **Teste:** O DataLoader carregou um *batch* gerando os tensores:
  - `Waveforms shape: torch.Size([16, 1, 160000])`
  - `Targets shape: torch.Size([16, 234])`

## 4. Inicialização e Teste do MLflow
- Criado o script `src/test_mlflow.py` para instanciar o tracking do MLflow localmente.
- O diretório `mlruns/` foi gerado no projeto.
- **Teste:** Um experimento de teste (`BirdCLEF_2026_EDA_Setup`) foi criado com sucesso, logando parâmetros (como *Batch Size*) e métricas *dummy* para validar a comunicação.

### Próximos Passos
Tudo está pronto para iniciar a **Fase 2 (Pipeline Base SED + EfficientNet)**, onde converteremos os *chunks* de 160.000 pontos (5s) em Mel-Espectrogramas e os passaremos por uma `EfficientNet-B0`.

Você já pode visualizar o painel do MLflow rodando no seu terminal:
```bash
.venv\Scripts\mlflow ui
```
