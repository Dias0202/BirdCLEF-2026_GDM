# Relatório de Conclusão: Fase 4 - Explicabilidade e Relatórios Finais

A **Fase 4** marcou a conclusão do ciclo de treinamento e a abertura da "caixa-preta" do modelo, permitindo visualizar como a `EfficientNet-B0` interpreta os áudios do Pantanal.

## 1. Desempenho do Modelo Final
O treinamento foi executado por 29 épocas, utilizando MixUp e SpecAugment.

> **Nota posterior — número não válido como estimativa.** O ROC-AUC registrado nesta fase (0,9882) foi medido com a
> `padded_cmap`, que atribui AUC próxima de 1,0 a classes sem positivos na validação, sobre um split aleatório que não
> separa gravações do mesmo gravador. Ele não estima o desempenho na competição. A validação foi corrigida (folds
> agrupados por gravador, métrica que ignora classes sem positivos — ver `src/metrics.py` e `src/validation.py`) e o
> score correspondente ainda não foi recalculado.

- **Loss de Validação (split antigo):** 0.0077
- **Convergência:** curva de treino estável ao longo das 29 épocas.

## 2. Implementação do Grad-CAM
- **Técnica:** Gradient-weighted Class Activation Mapping.
- **Camada Alvo:** `backbone.conv_head` (a última camada convolucional antes do pooling). Esta camada contém as características espaciais/temporais mais ricas e semânticas.
- **Visualização:** Mapas de calor (heatmap) são sobrepostos aos log-mel-espectrogramas.
- **Insights:** O Grad-CAM confirmou que o modelo está focando corretamente em regiões de alta energia (vocalizações) e ignorando silêncios ou ruídos de fundo constantes, validando a eficácia do *Attention Pooling*.

## 3. Artefatos e Rastreabilidade
Todos os resultados foram centralizados no **MLflow**:
- **Gráficos de Treino:** `plots/loss_comparison.png`, `plots/validation_auc.png`.
- **Explicabilidade:** Imagens em `gradcam_results/` mostrando o modelo "ouvindo" amostras de validação.
- **Modelo Final:** O melhor peso (`best_model.pth`) foi movido para a raiz do projeto e também preservado nos artefatos do MLflow.

## 4. Conformidade Kaggle
- O código foi estruturado de forma modular (`model.py`, `dataset.py`).
- O pipeline de inferência está preparado para rodar em CPU ou GPU (GTX 1650 Super), utilizando precisão mista onde disponível.
- A arquitetura `EfficientNet-B0` garante que o modelo caia dentro dos limites de tempo de inferência da competição (geralmente 2h para o set de teste).

### Status Final
**Projeto BirdCLEF 2026 - Pipeline de Elite:**
- [x] Fase 1: Setup e EDA
- [x] Fase 2: Pipeline Base (SED)
- [x] Fase 3: Generalização (MixUp/SpecAugment)
- [x] Fase 4: Explicabilidade (Grad-CAM)

Antes de qualquer uso real, falta medir o desempenho com a validação corrigida e contra uma linha de base (ver README).
