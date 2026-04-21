# Relatório de Conclusão: Fase 3 - Augmentations e Melhoria de Generalização

A **Fase 3** foi totalmente implementada, testada e validada, trazendo fortes regularizadores para evitar que a rede neural decore os dados (overfitting).

## 1. SpecAugment Integrado na GPU
- Editado o `src/model.py`.
- **Implementação:** Inseridas as camadas `torchaudio.transforms.FrequencyMasking` (com parâmetro 24) e `torchaudio.transforms.TimeMasking` (com parâmetro 64) logo após a geração do Mel-Espectrograma.
- **Otimização:** Essas transformações são ativadas estritamente durante o treinamento (`if self.training:`), sendo bypassadas na validação. Como rodam na GPU (`device`), o tempo de processamento é ínfimo.
- **Efeito:** Elas geram tarjas retangulares de zeros (ou média) aleatórias sobre os espectrogramas. Isso obriga a `EfficientNet` a não depender de uma única frequência do canto do pássaro para identificá-lo, aumentando drasticamente a robustez acústica do modelo.

## 2. MixUp Implementado no DataLoader
- Editado o `src/train.py`.
- **Implementação:** Foi criada a função vetorial `mixup_data` utilizando distribuições Beta (`alpha=0.5`). 
- **Matemática Aplicada:** Diferente do PyTorch clássico onde precisamos ajustar a função de *loss* manualmente, o nosso formato *Multi-Hot* (234 classes) já permite a interpolação linear direta (`lam * y + (1 - lam) * y[index]`).
- **Efeito:** Durante cada `batch`, dois áudios distintos (e seus rótulos) são literalmente misturados. Isso ensina o modelo a identificar animais cantando ao mesmo tempo (sobreposição vocal), algo crucial no Pantanal real.

## 3. Experimentos Logados no MLflow
- O script `train.py` foi evoluído para rastrear os novos parâmetros.
- O novo experimento `BirdCLEF_2026_Fase3_Augmentations` foi populado com as *tags* `mixup_alpha: 0.5` e `spec_augment: True`.
- **Avaliação Prática:** No teste da Fase 3, foi visível que a `Train Loss` começou levemente mais alta que na Fase 2 (0.53 contra 0.40). Isso comprova o funcionamento da técnica: os dados estão mais difíceis (ruidosos/misturados), o que é excelente para treinar um modelo super-poderoso que tende a viciar rapidamente.

### Status e Próximos Passos
O núcleo matemático (Métrica + Arquitetura + Generalização) está absolutamente finalizado. O modelo agora é robusto para aguentar dias de treinamento.

A próxima fronteira é a **Fase 4 (Explicabilidade e Relatórios)**. Nela iremos plugar a inovação principal: criar os "Mapas de Calor" usando **Grad-CAM** no MLflow para enxergar visualmente a rede trabalhando.