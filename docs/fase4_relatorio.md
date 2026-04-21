# Relatório de Conclusão: Fase 4 - Explicabilidade e Inovação Visual (Grad-CAM)

A **Fase 4** foi o grande diferencial deste projeto de detecção acústica. Conseguimos com sucesso implementar e integrar a técnica de Grad-CAM aos Espectrogramas gerados pela rede.

## 1. Implementação do Algoritmo Grad-CAM
- Foi criado o arquivo `src/gradcam.py`.
- **Dinâmica de Hooks:** O código aproveita os "hooks" (ganchos) nativos do PyTorch (`register_forward_hook` e `register_full_backward_hook`). Estes ganchos são acoplados à última camada convolucional da nossa `EfficientNet-B0` (camada identificada como `conv_head`).
- **Mecânica Matemática:** 
  - Durante o *forward pass* (quando o áudio entra), a rede salva os *Feature Maps* (as características brutas detectadas).
  - Durante o *backward pass* (quando calculamos o erro), a rede captura o Gradiente em relação à classe alvo.
  - Multiplicamos as características pelos gradientes passados por uma função ReLU, gerando assim um "Mapa de Calor" de onde a rede obteve mais ativação.

## 2. Visualização sobre o Mel-Espectrograma
- O módulo conta com a função `plot_gradcam` que faz uso da biblioteca `opencv-python` e `matplotlib`.
- Como o "Mapa de Calor" sai da CNN num tamanho muito pequeno (devido às reduções e *poolings*), nós o redimensionamos matematicamente para o tamanho real do Mel-Espectrograma inicial (ex: `128 bins x 313 frames`).
- A imagem resultante exibe o espectrograma colorido com o mapa de calor sobreposto, destacando os cantos dos pássaros que a rede "ouviu".

## 3. Integração ao Pipeline e Rastreamento de Artefatos no MLflow
- O script `src/test_gradcam.py` foi responsável pela prova de conceito.
- O modelo foi instanciado, um áudio real da competição foi processado e seu Grad-CAM foi salvo na imagem `gradcam_sample.png`.
- O experimento do MLflow `BirdCLEF_2026_Fase4_GradCAM` foi criado e a imagem foi salva estaticamente no diretório de "Artifacts" através da função `mlflow.log_artifact()`.

### Status e Próximos Passos
Temos um pipeline visual inovador e transparente. Especialistas biológicos agora podem não apenas ver a "predição" do modelo, mas confirmar visualmente qual pio exato na linha do tempo disparou a ativação para determinada espécie.

A última etapa é a **Fase 5 (Submissão e Otimização para Inferência)**, onde criaremos a lógica para iterar nos arquivos do diretório escondido de teste e produzir o `submission.csv` obedecendo o limite de execução sem internet do Kaggle.