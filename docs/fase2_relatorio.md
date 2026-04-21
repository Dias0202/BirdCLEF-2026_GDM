# Relatório de Conclusão: Fase 2 - Pipeline Base (SED + EfficientNet)

A **Fase 2** foi completamente implementada, testada e validada de acordo com o plano de implementação.

## 1. Arquitetura do Modelo (SEDModel)
- Criado o arquivo `src/model.py`.
- **Pré-processamento (Frontend):** Implementado diretamente na GPU dentro do módulo PyTorch utilizando `torchaudio.transforms.MelSpectrogram` e `AmplitudeToDB`. Isso economiza I/O e RAM do sistema, passando a responsabilidade de processamento paralelo para a GPU de 4GB, que dá conta de transformar *chunks* de 5 segundos rapidamente.
- **Backbone CNN:** Utilizamos a `EfficientNet-B0` da biblioteca `timm` (com pesos pré-treinados no ImageNet). Os classificadores originais foram removidos para isolar o extrator de características (*feature extractor*).
- **Attention Pooling (Mecanismo Temporal):** Foi implementada a classe `AttentionPooling`. Ela colapsa a dimensão temporal gerada pela EfficientNet, prestando atenção nos instantes em que o evento acústico de fato acontece. Uma camada Linear final mapeia o vetor para as **234 classes** do Pantanal.

## 2. Métrica Customizada
- Criado o script `src/metrics.py` com a função `padded_cmap`.
- Esta função replica a métrica *macro ROC AUC* do Kaggle, lidando com segurança matemática contra *batches* sem rótulos positivos (adicionando padronização `padding_factor`) e garantindo o comportamento avaliado na competição BirdCLEF.

## 3. Loop de Treinamento e Validação com AMP
- Implementado em `src/train.py`.
- **Mitigação para a GTX 1650 Super (4GB VRAM):** Integradas as ferramentas `torch.cuda.amp.autocast` e `GradScaler`. Isso garante que o *forward* e o *backward pass* operem em FP16 (*Half-Precision*), permitindo utilizar o tamanho de *batch* de 16 com folga e até dobrar a velocidade de treinamento na placa.
- Os *Dataloaders* da Fase 1 foram integrados perfeitamente.

## 4. Rastreamento e Logs (MLflow)
- O pipeline foi injetado com blocos do MLflow.
- No teste de homologação do pipeline (treinamento sobre um *micro-dataset* para garantir a integridade), os seguintes dados foram registrados com sucesso no *Tracking URI*:
  - **Parâmetros:** `model="EfficientNet-B0_AttentionPooling"`, `batch_size=16`, `optimizer="Adam"`, `learning_rate=1e-3`, `mixed_precision=True`.
  - **Métricas Reais:** `train_loss`, `val_loss`, e a mais importante, `val_auc` utilizando a métrica ROC-AUC baseada nas diretrizes oficiais.

### Status e Próximos Passos
O pipeline básico (de ponta a ponta) está vivo. O áudio cru entra, vira Mel-Espectrograma, passa pela rede, treina e salva métricas.
A próxima etapa é a **Fase 3 (Augmentations e Melhoria de Generalização)** para injetar `MixUp` e `SpecAugment` na *batch* e evitar o provável *overfitting* de um modelo poderoso como a EfficientNet.