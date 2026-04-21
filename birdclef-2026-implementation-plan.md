# Plano de Implementação: BirdCLEF 2026 - Acoustic Species Identification

## 1. Visão Geral e Objetivo
O objetivo é desenvolver um pipeline de Machine Learning de alto desempenho e eficiente para identificar espécies da fauna no Pantanal através de gravações sonoras (Soundscapes), de acordo com as regras da competição BirdCLEF+ 2026. A abordagem será rigorosamente otimizada para o hardware disponível sem uso de computação em nuvem paga.

## 2. Hardware e Limitações
- **Processador:** Intel i7-10700F
- **RAM:** 32GB DDR4
- **GPU:** NVIDIA GTX 1650 Super (4GB VRAM)

### 2.1 Estratégias de Mitigação de Hardware (GPU 4GB VRAM)
- **Automatic Mixed Precision (AMP):** O uso de FP16 (Half-Precision) no PyTorch reduzirá drasticamente o consumo de VRAM e acelerará os cálculos.
- **Micro-Batching:** Tamanhos de lote (Batch Size) conservadores (ex: 16 ou 32) associados ao *Gradient Accumulation*, se necessário simular batches maiores.
- **Data Loading:** Workers do PyTorch (`num_workers`) ajustados para paralelizar o I/O na CPU (aproveitando os núcleos do i7) sem engargalar a GPU. A extração inicial de Mel-Espectrogramas pode ser feita na GPU (usando `torchaudio` ou `nnAudio`) on-the-fly para poupar espaço em disco e I/O.

## 3. Stack Tecnológico e Frameworks
- **Deep Learning:** PyTorch (torchaudio para processamento, `timm` para modelos pré-treinados).
- **Rastreamento de Experimentos:** MLflow (Métricas, Parâmetros e Artifacts como os plots do Grad-CAM).
- **Data Science Básica:** Pandas, Numpy, Scikit-learn (para validação cruzada cruzada estratificada).
- **Explicabilidade:** Captum (PyTorch) ou implementações customizadas de Grad-CAM.

## 4. Arquitetura do Modelo (SED - Sound Event Detection)
O problema será tratado usando o paradigma *Sound Event Detection* adaptado para *Weakly Supervised Learning*, uma vez que as anotações principais indicam apenas a presença da ave no clipe (não o tempo exato em todos os dados).

- **Pré-processamento Front-end:** Transformar pedaços de áudio de 5 segundos em Mel-Espectrogramas logarítmicos.
- **Backbone CNN:** `EfficientNet-B0` (via biblioteca `timm`). Uma escolha excelente de *trade-off* entre acurácia e consumo de VRAM.
- **Cabeçote de Detecção (Pooling de Atenção):** 
  - As CNNs extraem características em mapas 2D (Tempo x Frequência). 
  - Utilizaremos *Attention Pooling* para condensar a dimensão de tempo em um único vetor, pesando frames mais importantes (onde a ave canta) com maiores valores de atenção.
  - Uma camada linear final classifica as 234 classes do Pantanal.

## 5. Augmentation de Dados (Específica para Áudio)
Para prevenir overfitting e garantir generalização:
- **SpecAugment:** Mascaramento aleatório de blocos de tempo e de frequência diretamente no Mel-Espectrograma.
- **MixUp:** Misturar dois exemplos de áudio e seus rótulos (*labels* multi-hot) para suavizar a fronteira de decisão.
- **Background Noise Addition:** Adicionar ruídos rosa/branco ou áudios classificados sem os animais (nocall) para treinar a robustez.

## 6. Métrica e Validação
- **Validação:** Validação Cruzada Estratificada (Stratified K-Fold) com 5 folds, usando `primary_label`.
- **Métrica Alvo:** c-macro ROC AUC (Métrica oficial do Kaggle para a competição, ignorando classes não presentes no conjunto).
- **Função de Perda:** BCEWithLogitsLoss (Binary Cross Entropy) visto que é um problema de classificação Multi-Label.

## 7. Explicabilidade e Inovação Visual (Grad-CAM)
O grande diferencial de explicabilidade da solução.
- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Será extraído o gradiente da última camada convolucional da EfficientNet para gerar um "mapa de calor".
- **Visualização:** Esse mapa de calor será sobreposto ao Mel-Espectrograma original.
- **Validação com MLflow:** Durante a validação a cada *epoch*, imagens do Grad-CAM geradas para áudios complexos serão salvas e logadas como artefatos no painel do MLflow. Isso permitirá "ver o que o modelo está ouvindo".

## 8. Fases de Implementação

### Fase 1: Setup do Ambiente e EDA
1. Configuração do ambiente PyTorch + CUDA compatível com a GTX 1650.
2. Inicialização e teste do MLflow Dashboard local.
3. Leitura e processamento das metadatas (`train.csv`, `taxonomy.csv`).
4. Criação do Dataset/DataLoader customizado em PyTorch focado em carregar e cortar *chunks* de 5 segundos.

### Fase 2: Pipeline Base (SED + EfficientNet)
1. Implementação da conversão para Mel-Espectrograma (on-the-fly).
2. Construção da classe do modelo SED baseada em `EfficientNet-B0` com Attention Pooling.
3. Escrita do loop de Treinamento e Validação integrando `torch.cuda.amp` (Mixed Precision).
4. Acoplamento de logs no MLflow (loss de treino, ROC-AUC de validação).

### Fase 3: Augmentations e Melhoria de Generalização
1. Integração de `MixUp` na batch de treinamento.
2. Adição de `SpecAugment`.
3. Hyperparameter tuning guiado por observação no MLflow (Learning Rate, taxas de dropout, peso do mixup).

### Fase 4: Explicabilidade e Relatórios
1. Implementação das rotinas do Grad-CAM.
2. Logging sistemático de imagens de validação (Spectrograma vs Grad-CAM) no MLflow Artifacts.
3. Exportação do código fonte para conformidade com regras do Kaggle.

### Fase 5: Submissão e Otimização para Inferência
1. Adaptação do pipeline para ler os `test_soundscapes` e produzir o `submission.csv` no tempo limite estipulado pelo Kaggle.
2. Criação do notebook final sem acesso à internet respeitando todas as `Code Requirements` da plataforma Kaggle.
