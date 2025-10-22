# Apostila – *Audio Transformers: Fundamentos e Aplicações*

## 1. Introdução ao Processamento de Áudio

O processamento de áudio é uma das áreas mais antigas e, ao mesmo tempo, mais dinâmicas da engenharia de sinais. Ele abrange desde a manipulação de ondas sonoras em tempo real até a extração de representações de alto nível capazes de alimentar modelos de inteligência artificial. Nesta seção, exploramos os fundamentos físicos e computacionais do som, o histórico das técnicas de análise e a motivação para o uso de **Transformers** em tarefas acústicas modernas.

---

### 1.1 O som como sinal

O **som** é uma variação de pressão que se propaga em um meio elástico, como o ar, a água ou um sólido. Essa variação é captada por microfones e convertida em um **sinal elétrico** ou digital, permitindo sua análise e manipulação.

#### a) Frequência e amplitude
- **Frequência (Hz)** representa o número de ciclos por segundo de uma onda sonora e está relacionada à percepção de **altura** (grave ou agudo).  
- **Amplitude** corresponde à magnitude da variação de pressão e está associada à **intensidade sonora** ou volume.  
Juntas, elas definem a forma de onda básica de qualquer som.

#### b) Espectro e domínio do tempo
Um sinal pode ser analisado em dois domínios:
- **Domínio do tempo**: representa a variação da amplitude ao longo do tempo (forma de onda).  
- **Domínio da frequência**: mostra a energia distribuída entre diferentes frequências (espectro).  
Transformações matemáticas como a **Transformada de Fourier (FFT)** permitem converter o sinal temporal em espectral.

#### c) Escalas perceptuais
Modelos perceptuais como a **escala Mel** ou a **Transformada Wavelet** ajustam a representação do espectro para se aproximar da percepção auditiva humana, enfatizando bandas de frequência mais relevantes para o ouvido.

#### d) Taxa de amostragem e quantização
Para representar o som no computador, ele deve ser **amostrado** — isto é, medido em intervalos regulares (tipicamente 16 kHz, 22,05 kHz ou 44,1 kHz).  
A **quantização** converte a amplitude analógica em valores discretos (8, 16 ou 24 bits).  
Segundo o **Teorema de Nyquist**, a taxa de amostragem deve ser ao menos o dobro da frequência máxima do sinal para evitar aliasing.

---

### 1.2 Da análise clássica aos modelos de aprendizado profundo

Durante décadas, o processamento de áudio baseou-se em **engenharia manual de atributos**, também chamada de *handcrafted features*. Essas abordagens utilizavam transformações matemáticas e estatísticas sobre o sinal.

#### a) Extração de *features* manuais
Entre as mais comuns:
- **MFCC (Mel-Frequency Cepstral Coefficients)**: representam o envelope espectral em uma escala perceptual Mel.  
- **ZCR (Zero Crossing Rate)**: mede a taxa de cruzamentos por zero e indica se o som é suave ou ruidoso.  
- **Energia**: quantifica a potência média do sinal.  
- **Entropia espectral**: indica o grau de desordem na distribuição de energia das frequências.  

Essas *features* são compactas e fáceis de calcular, sendo amplamente usadas em tarefas como reconhecimento de fala e identificação de locutor.

#### b) Limitações de modelos baseados em características fixas
Embora eficientes, essas representações:
- Dependem fortemente de escolhas de parâmetros manuais.  
- Não capturam bem o contexto temporal de longa duração.  
- Sofrem degradação sob ruído, reverberação e variações de fala.  

Assim, elas não são ideais para aplicações complexas como reconhecimento de emoção, *speaker diarization* ou detecção de falsificações (*deepfakes*).

#### c) A transição para *representation learning*
Com o avanço das redes neurais profundas, a ênfase passou da engenharia manual para o **aprendizado de representações automáticas**.  
Modelos como CNNs e RNNs passaram a aprender *features* diretamente dos espectrogramas ou do próprio áudio bruto, reduzindo a dependência de pré-processamento manual.

---

### 1.3 Motivação para Transformers em áudio

O surgimento dos **Transformers** revolucionou o processamento de linguagem natural e, posteriormente, o de áudio. Sua arquitetura baseada em **atenção** permite modelar relações de longo alcance sem depender de estruturas sequenciais estritas.

#### a) Desafios no processamento de áudio
- **Contexto longo**: sons podem durar vários segundos ou minutos, com dependências distantes no tempo.  
- **Multimodalidade**: muitas tarefas combinam áudio, texto e vídeo.  
- **Generalização**: diferentes falantes, idiomas e ruídos exigem modelos robustos.  

Transformers são ideais para lidar com esses aspectos, pois tratam o áudio como uma sequência de embeddings vetoriais e aprendem relações entre eles sem restrições de vizinhança local.

#### b) Por que CNNs e RNNs são insuficientes
- **CNNs** capturam apenas relações locais, dificultando o entendimento de dependências temporais longas.  
- **RNNs** e **LSTMs** sofrem com *vanishing gradients* e paralelização limitada.  
Transformers, por outro lado, processam todas as posições da sequência simultaneamente, utilizando o mecanismo de atenção para determinar quais partes do sinal são relevantes em cada instante.

#### c) Abordagem *self-supervised* e *transfer learning*
Em vez de depender de rótulos manuais, os Transformers de áudio modernos são treinados de forma **auto-supervisionada** — ou seja, aprendem a prever partes mascaradas do sinal.  
Após esse pré-treinamento em grandes corpora de áudio, podem ser ajustados (*fine-tuned*) para tarefas específicas, como:
- Reconhecimento de fala (ASR)  
- Identificação de emoção  
- Detecção de falsificações (deepfakes)  
- Classificação de sons ambientais  

Essa abordagem, conhecida como **transfer learning**, permite que modelos pré-treinados sejam reutilizados de forma eficiente em novos contextos, economizando tempo e dados.

---

## 2. Fundamentos dos Transformers

Os **Transformers** revolucionaram o aprendizado de máquina ao substituir estruturas recursivas e convolucionais pelo mecanismo de **atenção**, capaz de capturar dependências de longo alcance entre elementos de uma sequência.  
Originalmente proposto por Vaswani et al. (2017) no artigo *“Attention is All You Need”*, o modelo tornou-se a base de diversas arquiteturas modernas, tanto em texto quanto em áudio, imagem e multimodalidade.

---

### 2.1 Arquitetura geral

O Transformer é composto por duas partes principais: **Encoder** e **Decoder**, que podem ser usados juntos (como em tradução automática) ou isoladamente (como em classificação, reconhecimento de fala ou codificação de embeddings de áudio).

#### a) Encoder
O **encoder** recebe uma sequência de vetores de entrada (por exemplo, frames de áudio ou tokens de texto) e gera representações internas contextualizadas.  
Cada camada do encoder contém:
- Um bloco de **multi-head self-attention**, que aprende as relações entre todos os elementos da sequência.  
- Um bloco **feed-forward**, que refina as representações de cada posição.  
Ambos os blocos utilizam **residual connections** e **layer normalization** para facilitar o treinamento.

#### b) Decoder
O **decoder** é responsável por gerar saídas sequenciais (como texto em ASR ou legendas de áudio). Ele também contém camadas de auto-atenção e atenção cruzada (cross-attention), permitindo “enxergar” tanto o contexto anterior da saída quanto o do encoder.

#### c) Atenção (*self-attention*) e *multi-head attention*
O núcleo do Transformer é o mecanismo de **auto-atenção**, que permite que cada elemento da sequência pese a importância dos outros.  
Em vez de processar sequências de forma estritamente temporal, o Transformer aprende **quais partes do sinal são mais relevantes** para cada instante.

O modelo utiliza múltiplas cabeças de atenção (*multi-head attention*) para aprender diferentes tipos de relacionamentos em paralelo — por exemplo, timbre, prosódia ou ritmo em sinais de voz.

#### d) Embeddings posicionais
Como o Transformer não tem estrutura sequencial explícita, ele precisa de uma maneira de representar a **posição temporal** de cada elemento.  
Isso é feito por meio dos **positional embeddings**, que são vetores adicionados às entradas originais e codificam informações sobre a ordem dos frames.

#### e) Normalização e residual connections
Para estabilizar o aprendizado e permitir treinamento profundo, o Transformer usa:
- **Layer Normalization**: normaliza cada vetor de entrada por dimensão.  
- **Residual connections**: adicionam a entrada original ao resultado da camada, evitando perda de informação.

> **Figura sugerida:** diagrama mostrando o fluxo Encoder → Multi-Head Attention → Feed-Forward → Normalization.

---

### 2.2 Fórmulas essenciais

O mecanismo de atenção pode ser descrito matematicamente como:

```math
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

onde:

- **Q (Queries)**: representa o vetor que busca informações relevantes.  
- **K (Keys)**: representa os vetores que podem conter as informações buscadas.  
- **V (Values)**: contém os valores associados às chaves.  
- **dₖ**: é a dimensão dos vetores K (usada para normalização numérica).

O produto `QKᵀ` mede a **similaridade** entre cada par de vetores — análogo ao cosseno entre eles.  
O **softmax** converte esses valores em pesos de atenção, que determinam **quais partes do sinal devem receber mais foco**.  
Multiplicando por `V`, o modelo combina as informações ponderadas e gera uma **nova representação contextualizada**.

---

### a) Atenção multi-cabeça

Na prática, são usados diversos conjuntos de `(Q, K, V)` simultaneamente — cada um chamado de **cabeça de atenção** (*attention head*).  
Isso permite que o modelo aprenda **relações complementares** em diferentes subespaços de representação, como padrões rítmicos, harmônicos e fonéticos.

Cada cabeça aprende um aspecto distinto da estrutura temporal ou espectral do sinal, e suas saídas são concatenadas e projetadas novamente em um único espaço vetorial.

---

### b) Interpretação intuitiva

Em áudio, o mecanismo de atenção permite, por exemplo:

- Destacar partes do sinal que contêm **vogais ou consoantes específicas**;  
- Aprender a **relação entre sílabas distantes no tempo**;  
- Focar em **regiões de energia ou ruído** relevantes para a tarefa de fala ou emoção.  

Essas propriedades tornam os Transformers capazes de **compreender contexto acústico global**, uma limitação significativa em redes puramente convolucionais ou recorrentes.

---

## 2.3 Pré-treinamento e *fine-tuning*

A força dos Transformers está no uso de **pré-treinamento em larga escala**, seguido de uma **adaptação leve** (*fine-tuning*) para tarefas específicas.  
O pré-treinamento fornece ao modelo conhecimento geral sobre o domínio acústico, que depois é refinado com um conjunto menor e rotulado de dados.

---

### a) Máscaras de previsão (MLM, CTC)

- **MLM (Masked Language Modeling)**: o modelo aprende a prever partes mascaradas da entrada — em áudio, trechos de tempo são ocultados e o modelo tenta reconstruí-los. Essa técnica ensina o Transformer a **entender dependências temporais e estruturais**.  
- **CTC (Connectionist Temporal Classification)**: usada em reconhecimento de fala, associa trechos do sinal a sequências de texto **sem necessidade de alinhamento explícito**, permitindo treinar modelos de ASR de ponta a ponta.

---

### b) Congelamento parcial de camadas

Durante o *fine-tuning*, é comum **congelar** as primeiras camadas do modelo — responsáveis por representar **padrões acústicos gerais**, como espectro e ritmo — e treinar apenas as últimas, que aprendem **informações específicas da tarefa** (emoção, patologia, idioma, etc.).  

Essa prática:
- Reduz o custo computacional;  
- Evita *overfitting*;  
- Aproveita o conhecimento aprendido durante o pré-treinamento.

---

### c) Estratégias de adaptação

Diversas técnicas modernas permitem adaptar grandes Transformers **sem treinar todos os parâmetros originais**, tornando o processo mais leve e eficiente:

- **Linear Probing**: utiliza apenas as representações pré-treinadas como entrada para um novo classificador linear (por exemplo, uma camada densa).  
- **LoRA (Low-Rank Adaptation)**: insere pequenas matrizes de baixo posto nas camadas de atenção, ajustando apenas uma fração dos pesos originais.  
- **Adapters**: adicionam blocos leves dentro das camadas do Transformer, permitindo **adaptação modular** entre tarefas sem alterar o modelo base.

Essas estratégias viabilizam o uso de **modelos gigantes em aplicações reais**, como reconhecimento de fala embarcado, dispositivos móveis e sistemas de **detecção de deepfakes**, mantendo alta precisão com custo reduzido.

---

## 3. Pré-processamento de Áudio

O pré-processamento é uma etapa essencial antes de alimentar sinais de áudio em modelos de aprendizado profundo.  
Seu objetivo é **padronizar as amostras**, **garantir consistência temporal** e **extrair representações numéricas** adequadas ao treinamento.

Na prática, diferentes gravações podem apresentar taxas de amostragem, durações e volumes distintos.  
Sem uma padronização, o modelo pode aprender padrões artificiais relacionados à forma de captura — e não às propriedades acústicas reais do som.

---

### 3.1 Padronização

O primeiro passo é preparar o sinal para garantir compatibilidade entre todas as amostras.

#### a) Conversão para mono, 16 kHz, PCM 16-bit
- **Mono:** converte sinais estéreo (duas faixas) em um único canal, normalmente pela média dos canais esquerdo e direito. Isso reduz a dimensionalidade e garante uniformidade.  
- **16 kHz:** é uma taxa de amostragem suficiente para fala humana, pois cobre a faixa de frequências até 8 kHz, onde se concentram a maioria dos formantes e sons vocais.  
- **PCM 16-bit:** representa cada amostra como um inteiro de 16 bits (−32.768 a 32.767), balanceando precisão e tamanho de arquivo.

> A padronização facilita o uso de *pretrained models*, que geralmente esperam entradas em mono e 16 kHz (por exemplo, Wav2Vec2, HuBERT e Whisper).

#### b) Normalização de amplitude
Sons gravados em diferentes ambientes podem ter níveis de volume distintos.  
A **normalização** ajusta a amplitude do sinal para uma faixa comum (geralmente entre −1.0 e +1.0), evitando saturação e melhorando a estabilidade numérica durante o treinamento.

```python
import numpy as np
y = y / np.max(np.abs(y))
```

#### c) Segmentação (*chunking*) e *padding*
Modelos Transformers exigem entradas com **comprimento fixo**.  
Por isso, o áudio é dividido em **segmentos menores** (por exemplo, 2 ou 4 segundos).  
Quando um segmento é menor que o tamanho desejado, aplica-se **padding**, preenchendo o restante com zeros.

```python
import librosa
import numpy as np

y, sr = librosa.load("voz.wav", sr=16000)
segment_length = int(sr * 4)  # 4 segundos

# Segmenta o áudio em blocos de 4s
chunks = [y[i:i+segment_length] for i in range(0, len(y), segment_length)]

# Adiciona padding se o último bloco for menor
if len(chunks[-1]) < segment_length:
    chunks[-1] = np.pad(chunks[-1], (0, segment_length - len(chunks[-1])))
```

Essa etapa é especialmente importante em **modelos de inferência por lote (batch)**, que exigem vetores de tamanho igual.

---

### 3.2 Visualização com `librosa`

A biblioteca **Librosa** é amplamente utilizada para manipulação e análise de sinais de áudio em Python.  
Ela permite tanto o carregamento quanto a visualização de formas de onda, espectrogramas e transformadas diversas.

O exemplo abaixo mostra como carregar um arquivo `.wav`, padronizar sua taxa de amostragem para 16 kHz e exibir a **forma de onda temporal**.

```python
import librosa, librosa.display, matplotlib.pyplot as plt

# Carrega o áudio e define a taxa de amostragem padrão
y, sr = librosa.load("sample.wav", sr=16000)

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title("Forma de onda - sample.wav")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
```

#### Interpretação:
- As regiões de maior amplitude indicam sons mais intensos.  
- Trechos silenciosos aparecem como linhas próximas de zero.  
- Essa visualização é útil para inspecionar ruídos, pausas e cortes abruptos.

---

### 3.3 Espectrogramas e transformadas

O **espectrograma** é uma das representações mais importantes em processamento de áudio.  
Ele mostra como a energia do sinal se distribui ao longo do tempo e da frequência.

A função `librosa.feature.melspectrogram` calcula o espectrograma na **escala Mel**, mais próxima da percepção auditiva humana.

```python
import numpy as np
import matplotlib.pyplot as plt

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr)
plt.title("Mel-Espectrograma")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()
```

#### Explicação:
- `n_mels=128` define o número de bandas na escala Mel (128 é um valor comum).  
- A função `power_to_db` converte a escala linear em decibéis, tornando os contrastes mais visíveis.  
- O eixo **y** mostra as bandas de frequência perceptual, enquanto o eixo **x** mostra o tempo.

> **Figura sugerida:** comparação entre **waveform**, **espectrograma linear** e **mel-espectrograma**, destacando como cada representação enfatiza aspectos diferentes do mesmo sinal.

---

#### Comparação entre representações:
| Representação | Enfatiza | Uso típico |
|---------------|-----------|------------|
| **Forma de onda** | Amplitude no tempo | Visualização e inspeção manual |
| **Espectrograma linear** | Energia por frequência | Análise de ruído, timbre |
| **Mel-espectrograma** | Frequências perceptuais | Entrada para CNNs e Transformers |

---

O pré-processamento adequado é crucial para o desempenho dos modelos baseados em Transformers.  
Pequenas inconsistências de taxa de amostragem, amplitude ou duração podem gerar **falhas no alinhamento temporal** e comprometer a qualidade dos embeddings extraídos nas etapas seguintes.a 
---

## 4. Modelos de Transformers para Áudio

Os **Transformers aplicados ao áudio** representam uma das evoluções mais significativas do aprendizado de máquina em sinais sonoros.  
Eles substituíram arquiteturas tradicionais baseadas em convoluções (CNNs) e recorrência (RNNs) por modelos de atenção auto-supervisionados capazes de aprender **representações ricas e contextuais** de voz, fala e som ambiente — sem necessidade de rótulos manuais.

Nesta seção, apresentamos os principais modelos da área, suas diferenças e exemplos práticos de uso.

---

### 4.1 Linha do tempo dos principais modelos

Abaixo está uma linha do tempo com os principais **modelos Transformers de áudio**, destacando seu ano, arquitetura base, tipo de aprendizado e tarefa predominante:

| Modelo | Ano | Base | Tipo | Tarefa principal |
|--------|------|------|------|------------------|
| **Wav2Vec** | 2019 | CNN + Transformer | Self-supervised | Fala |
| **HuBERT** | 2021 | Transformer | Self-supervised | Fala |
| **WavLM** | 2022 | Transformer | Multi-task | Fala |
| **Whisper** | 2022 | Encoder–Decoder | ASR Multilíngue |
| **AudioMAE** | 2023 | Vision Transformer | Masked Audio Modeling |
| **CLAP** | 2023 | Dual Encoder | Áudio + Texto |

---

### a) Wav2Vec (Facebook AI, 2019)
O **Wav2Vec** foi um dos primeiros modelos a aplicar *self-supervised learning* diretamente sobre o **áudio bruto (waveform)**.  
Ele utiliza uma CNN para extrair *features* locais e um Transformer para modelar dependências de longo alcance.  
Durante o pré-treinamento, o modelo tenta prever partes mascaradas do sinal a partir do contexto.

Principais contribuições:
- Substituiu espectrogramas por **waveform direto**, eliminando pré-processamento pesado.  
- Aprendeu representações robustas a ruído e variações de fala.  
- Serviu de base para a segunda geração, **Wav2Vec 2.0**, amplamente utilizada.

---

### b) HuBERT (Hidden Unit BERT, 2021)
O **HuBERT** (da Meta AI) introduziu um método mais estável de pré-treinamento:  
em vez de prever amplitudes, ele tenta prever **rótulos ocultos** (clusters) de segmentos de áudio.

Etapas:
1. Um modelo simples de *k-means* gera rótulos de áudio brutos.  
2. O Transformer é treinado para prever o rótulo mascarado de cada segmento.  
3. As representações aprendidas são refinadas em múltiplas iterações.

Vantagens:
- Aprendizado mais estável e menos sensível a ruído.  
- Gera embeddings altamente discriminativos, ideais para **reconhecimento de fala** e **análise de emoções**.

---

### c) WavLM (Microsoft, 2022)
O **WavLM** amplia o HuBERT adicionando:
- **Tarefas múltiplas (multi-task learning)**, como separação de locutores e ruído;  
- **Dados misturados com ruído sintético** durante o pré-treinamento;  
- **Camadas adicionais** para melhor modelagem contextual.

Seu diferencial é a **robustez em ambientes ruidosos** e **melhor transferência para múltiplas tarefas**, como ASR, *speaker verification* e *emotion recognition*.  
Ele é considerado o estado da arte em *universal speech representation learning*.

---

### d) Whisper (OpenAI, 2022)
O **Whisper** foi treinado com **680 mil horas de áudio e transcrição** em múltiplos idiomas.  
É um modelo **encoder–decoder**, similar ao T5, que realiza *Automatic Speech Recognition (ASR)* de forma direta.

Características principais:
- Suporte multilíngue (mais de 90 idiomas).  
- Capaz de **tradução direta** e **detecção de idioma**.  
- Extremamente robusto a sotaques, ruídos e gravações imperfeitas.  
- Funciona tanto localmente quanto via *pipeline* da Hugging Face.

> O Whisper não é apenas um modelo de codificação: ele **gera texto** a partir de áudio, operando de forma autoregressiva.

---

### e) AudioMAE (Meta AI, 2023)
Inspirado no *Masked Autoencoder (MAE)* usado em visão computacional, o **AudioMAE** aplica o mesmo princípio ao áudio:
- Segmentos do sinal são **mascarados aleatoriamente**.  
- O modelo aprende a **reconstruí-los** a partir do contexto visível.  

Essa técnica melhora a compreensão de **padrões acústicos globais** e **relações tempo-frequência**.  
O AudioMAE é frequentemente usado como modelo de base para tarefas de **classificação de sons** e **eventos acústicos**.

---

### f) CLAP (Contrastive Language-Audio Pretraining, 2023)
O **CLAP**, desenvolvido pela LAION, é um modelo **multimodal** que aprende a alinhar **áudio e texto** em um mesmo espaço vetorial.  
Ele é treinado com milhões de pares (som, legenda) para aprender **representações contrastivas** — similares às do CLIP em visão.

Aplicações:
- *Audio Captioning* (geração de legendas de áudio).  
- Busca semântica (“encontre sons de passos” → retorna arquivos correspondentes).  
- Fusão texto-áudio para análise multimodal.

O CLAP marca o início da era **multimodal** em processamento de som, unificando percepção acústica e compreensão linguística.

---

### 4.2 Uso prático com Hugging Face

Os modelos de áudio baseados em Transformers estão disponíveis no **Hugging Face Hub**, permitindo acesso direto a arquiteturas pré-treinadas para diversas tarefas.

#### Instalação

```bash
pip install torch torchaudio transformers librosa
```

---

### Exemplo – Extração de *embeddings* com Wav2Vec2

O exemplo abaixo mostra como carregar o modelo **Wav2Vec2** e extrair embeddings a partir de um arquivo `.wav`.  
Esses embeddings podem ser usados em classificadores downstream (como SVM, XGBoost, ou redes densas).

```python
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, librosa, numpy as np

# Carrega o processador e o modelo pré-treinado
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Carrega o áudio padronizado (mono, 16 kHz)
y, sr = librosa.load("voz_exemplo.wav", sr=16000)

# Pré-processa e gera tensores de entrada
inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)

# Extrai embeddings sem gradientes
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

print(embeddings.shape)
```

#### Interpretação:
- `processor` converte o áudio em tokens normalizados para o modelo.  
- `model` gera embeddings de alta dimensão para cada frame.  
- `mean(dim=1)` obtém a média temporal, criando um vetor representativo do áudio inteiro.  

Esses vetores podem ser salvos e usados para:
- Treinar classificadores de emoção, idioma ou patologia;  
- Calcular similaridade acústica entre amostras;  
- Agrupar falas por locutor (*speaker clustering*).

---

> **Dica:** para tarefas como ASR (reconhecimento de fala), substitua `Wav2Vec2Model` por `Wav2Vec2ForCTC`, que inclui uma cabeça de predição de texto.

---

A partir dessa base, o próximo passo é **avaliar e comparar** os embeddings gerados por diferentes modelos em tarefas específicas, explorando métricas como AUC, EER e Brier — tema dos próximos capítulos.


## 5. Extração e Visualização de Embeddings

Embeddings de áudio são **vetores densos** que codificam informações acústicas e linguísticas em um espaço contínuo.  
Eles permitem alimentar classificadores leves (SVM, XGBoost), medir similaridade entre amostras e inspecionar separabilidade de classes.

> **Objetivo deste capítulo:** mostrar como extrair, reduzir a dimensionalidade e **visualizar** embeddings para análises qualitativas e quantitativas.

---

### 5.1 Redução de dimensionalidade

Como embeddings costumam ter **centenas** ou **milhares** de dimensões, usamos métodos de redução para projetá-los em 2D/3D, mantendo a estrutura local/global dos dados.

#### (a) Pipeline típico
1. **Extrair embeddings** do modelo (e.g., Wav2Vec2/HuBERT/WavLM).  
2. **Normalizar** (opcional, mas recomendado) – `StandardScaler` ou `normalize(L2)`.  
3. **Pré-reduzir com PCA** (opcional, acelera t-SNE/UMAP e remove ruído).  
4. **Projeção não linear** – UMAP ou t-SNE para visualização.  

#### (b) t-SNE (bom para vizinhança local)
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_embedded = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                  init='pca', random_state=42).fit_transform(embeddings)

plt.figure(figsize=(7,5))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, alpha=0.8)
plt.title("Visualização dos Embeddings de Áudio (t-SNE)")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.tight_layout()
plt.show()
```
**Dicas:**  
- `perplexity` (5–50) regula o “nº de vizinhos” efetivos; teste 30/40 para bases médias.  
- Use `init='pca'` e `random_state` fixo para reprodutibilidade.  
- Para _datasets_ grandes, faça **PCA→t-SNE** (ex.: PCA para 50D, depois t-SNE).

#### (c) UMAP (rápido, preserva estrutura global/local)
```python
# pip install umap-learn
import umap
import matplotlib.pyplot as plt

umap_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric="cosine", random_state=42)
U = umap_2d.fit_transform(embeddings)

plt.figure(figsize=(7,5))
plt.scatter(U[:, 0], U[:, 1], s=10, alpha=0.8)
plt.title("Visualização dos Embeddings de Áudio (UMAP)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
plt.show()
```
**Dicas:**  
- `n_neighbors` controla o balanço **local vs global** (menor = mais local).  
- `min_dist` controla a “compactação” dos grupos (menor = clusters mais densos).  
- `metric="cosine"` funciona bem para embeddings.

#### (d) PCA (linear, útil para pré-processar)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
X_50 = pca.fit_transform(embeddings)   # usar como entrada de t-SNE/UMAP
print("Variância explicada acumulada (50D):", pca.explained_variance_ratio_.sum())
```
**Uso típico:** reduzir ruído e acelerar t-SNE/UMAP.

---

### 5.2 Interpretação

A interpretação de projeções 2D/3D é **qualitativa**: buscamos padrões visuais que façam sentido para a tarefa.

- **Clusters** podem indicar **padrões fonéticos**, **timbre**, **emoção**, **idioma** ou **locutor**.  
- **Separabilidade** entre classes (e.g., **real vs fake**, **saudável vs patológico**) sugere que o espaço vetorial contém **sinais discriminativos** úteis para um classificador.  
- **Sobreposição** excessiva sinaliza necessidade de:  
  - Melhor **pré-processamento** (normalização, _chunking_);  
  - Outro **modelo de embeddings** (e.g., WavLM vs HuBERT);  
  - **Fusão** com *handcrafted features* (MFCC/CQCC) ou dados **multimodais**.

#### Visualização por classe (cores)
```python
import matplotlib.pyplot as plt
import numpy as np

# labels: array 1D com rótulos (ex.: 0=real, 1=fake)
classes = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

plt.figure(figsize=(7,5))
for c, col in zip(classes, colors):
    mask = (labels == c)
    plt.scatter(U[mask, 0], U[mask, 1], s=12, alpha=0.85, label=str(c), color=col)
plt.legend(title="Classe"); plt.title("Embeddings por Classe (UMAP)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout(); plt.show()
```

#### Medidas quantitativas de clusterização
Para complementar a análise visual, use métricas **sem rótulo** (se houver _pseudo-labels_) ou **com rótulo**:

- **Silhouette Score** (maior é melhor; −1 a 1).  
- **Davies–Bouldin Index** (menor é melhor).  
- **Calinski–Harabasz** (maior é melhor).

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print("Silhouette:", silhouette_score(U, labels, metric='euclidean'))
print("Davies–Bouldin:", davies_bouldin_score(U, labels))
print("Calinski–Harabasz:", calinski_harabasz_score(U, labels))
```

> **Nota:** Métricas de cluster avaliam **forma e separação** na projeção.  
> Para medir **poder preditivo**, treine um classificador e avalie (AUC, EER, F1, etc.).

---

### Boas práticas e armadilhas comuns

- **Balanceamento:** amostras muito desbalanceadas distorcem a projeção; amostre estratificadamente.  
- **Escala:** normalize embeddings (L2) antes de PCA/UMAP/t-SNE quando necessário.  
- **Reprodutibilidade:** fixe `random_state`/semente.  
- **Tamanho do dataset:** para coleções muito grandes, use **PCA incremental** ou amostragem.  
- **Evite vazamento:** se fará avaliação por *folds*, **não** ajuste UMAP/t-SNE no conjunto de teste.  
- **Múltiplas rodadas:** compare diferentes `perplexity` (t-SNE) e `n_neighbors` (UMAP) — mudanças grandes e instáveis podem indicar que a estrutura não é robusta.

---

### Checklist rápido

- [ ] Embeddings extraídos e **padronizados** (mono/16 kHz, _chunking_ consistente).  
- [ ] **Normalização** aplicada quando apropriado.  
- [ ] **PCA→(UMAP|t-SNE)** testados com hiperparâmetros principais.  
- [ ] Gráficos 2D com **cores por classe** e legendas legíveis.  
- [ ] **Métricas de cluster** calculadas para corroborar a inspeção visual.  
- [ ] Anotações sobre **insights** (clusters por locutor, emoção, idioma, real/fake).  

---

> **Resumo:** Reduzir e visualizar embeddings ajuda a **diagnosticar** se a representação captura os fatores de variação relevantes.  
> Combine análise visual + métricas de cluster + avaliação preditiva para tomar decisões sólidas sobre o modelo e o pré-processamento.
 

## 6. Aplicações Práticas

Este capítulo apresenta **exemplos reais de aplicação dos modelos de áudio baseados em Transformers**, cobrindo desde tarefas tradicionais de reconhecimento de fala até domínios mais recentes, como **detecção de deepfakes**, **patologias vocais** e **descrição multimodal de sons**.  
O foco é demonstrar como as representações aprendidas (embeddings) podem ser utilizadas de forma prática em diferentes contextos.

---

### 6.1 Reconhecimento Automático de Fala (ASR)

O **Reconhecimento Automático de Fala (Automatic Speech Recognition)** transforma áudio em texto.  
Modelos baseados em Transformers, como o **Whisper (OpenAI)**, tornaram essa tarefa mais robusta, multilíngue e acessível.

#### Exemplo prático usando Whisper
```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Transcrição automática
resultado = asr("fala_exemplo.wav")
print(resultado["text"])
```

#### Explicação:
- `pipeline` abstrai todo o processo de pré-processamento e decodificação.  
- O modelo "whisper-small" é leve e rápido, ideal para uso local.  
- Suporta mais de **90 idiomas** e pode até **traduzir fala** para o inglês automaticamente.

#### Aplicações:
- Legendas automáticas e acessibilidade.  
- Transcrição de podcasts, reuniões e palestras.  
- Pré-processamento para tarefas de **análise de sentimento**, **resumo de fala** e **busca semântica**.

> **Dica:** para longos áudios, divida o arquivo em trechos (chunking) de 30 segundos e combine as transcrições.

---

### 6.2 Detecção de Patologias Vocais

A **análise de voz patológica** é uma das aplicações clínicas mais promissoras dos Transformers.  
O objetivo é identificar automaticamente desvios vocais associados a disfonias, paralisias de pregas vocais, nódulos e outras condições.

#### Estratégia geral:
1. Extrair **embeddings fixos** (HuBERT, WavLM).  
2. Treinar um **classificador leve** (SVM, XGBoost ou MLP).  
3. Avaliar com métricas robustas (AUC, F1, EER, Brier).  

```python
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

y_true = [...]  # rótulos (0 = saudável, 1 = patológico)
y_pred = modelo.predict_proba(X_emb)[:,1]

print("AUC:", roc_auc_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred > 0.5))
print("Brier:", brier_score_loss(y_true, y_pred))
```

#### Observações clínicas:
- As variações sutis de **timbre**, **rugosidade** e **instabilidade harmônica** podem ser captadas pelos embeddings de Transformers.  
- Combinar embeddings com *features* acústicas (jitter, shimmer, HNR) tende a melhorar o desempenho.  
- Bases de dados como **MEEI Voice**, **SVD** e **AVPD** são comumente usadas em pesquisas.

---

### 6.3 Detecção de Deepfakes

A **detecção de deepfakes em voz** é uma aplicação crítica de segurança digital.  
Transformers podem capturar padrões sutis de geração artificial e inconsistências espectrais.

#### Pipeline típico:
1. **Extração de características híbridas:**
   - *Handcrafted features*: CQCC, MFCC, ZCR, Teager.  
   - *Embeddings*: WavLM, HuBERT, Whisper.  
   - Fusão precoce (*early fusion*) ou tardia (*late fusion*).

2. **Treinamento de modelos discriminativos:**
   - Classificadores: **XGBoost**, **Random Forest**, **SVM**.  
   - Calibração de probabilidades: **Platt Scaling** ou **Isotonic Regression**.

3. **Interpretação com XAI:**
   - Uso de **SHAP** para identificar as *features* mais relevantes.  
   - Visualização de padrões de atenção no espectrograma.

```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_emb)
shap.plots.beeswarm(shap_values)
```

#### Métricas recomendadas:
- **EER (Equal Error Rate):** mede equilíbrio entre falsos positivos e negativos.  
- **AUC (Area Under Curve):** qualidade geral da separação.  
- **ECE / Brier:** calibração das probabilidades de decisão.

#### Observações:
- Transformers capturam irregularidades temporais e espaciais criadas por vocoders.  
- A combinação *CQCC + WavLM* tem mostrado ótimo desempenho em benchmarks como **ASVspoof 2021**.  
- Ferramentas SHAP e Grad-CAM auxiliam na interpretabilidade dos resultados.

---

### 6.4 Audio Captioning e Retrieval

O **Audio Captioning** consiste em gerar **descrições textuais** para sons, enquanto o **Audio Retrieval** busca sons similares com base em texto.

Modelos multimodais como **CLAP** e **AudioCLIP** aprendem representações conjuntas entre **áudio e linguagem natural**, permitindo correspondência cruzada entre modalidades.

#### Exemplo prático com CLAP
```python
from transformers import ClapProcessor, ClapModel
import torch, librosa

# Carrega o modelo multimodal
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

# Entrada: áudio e texto
text = ["som de uma guitarra elétrica"]
y, sr = librosa.load("guitarra.wav", sr=48000)
inputs = processor(audios=y, texts=text, sampling_rate=sr, return_tensors="pt", padding=True)

# Similaridade entre áudio e texto
with torch.no_grad():
    outputs = model(**inputs)
    similarity = torch.cosine_similarity(outputs.audio_embeds, outputs.text_embeds)
print(f"Similaridade áudio-texto: {similarity.item():.3f}")
```

#### Aplicações práticas:
- **Sistemas de busca por som** ("encontre áudios com passos, chuva, risada").  
- **Geração de legendas automáticas** em vídeos e datasets.  
- **Organização semântica de bibliotecas sonoras**.  

> Modelos como CLAP e AudioCLIP inauguram uma nova era de **análise multimodal**, integrando visão, texto e som em um mesmo espaço de embeddings.

---

### Conclusão

As aplicações apresentadas demonstram a versatilidade dos Transformers em tarefas acústicas e multimodais.  
Do diagnóstico clínico à segurança digital, passando por transcrição e compreensão semântica, essas arquiteturas consolidam o áudio como um **domínio de aprendizado de representação de ponta**.  

O próximo passo é compreender como avaliar, calibrar e explicar os resultados obtidos — tema abordado nos capítulos seguintes.


---

## 7. Explicabilidade e Interpretação

Os modelos de áudio baseados em Transformers alcançam alto desempenho, mas sua **interpretação** é essencial em cenários sensíveis (clínica, segurança, educação). Nesta seção, mostramos como **inspecionar a atenção**, **explicar decisões** com XAI e **analisar a confiança ao longo do tempo**.

---

### 7.1 Mapas de Atenção

**O que são:** os mapas de atenção indicam **quanto** cada posição (frame/token) "olha" para as demais ao gerar a representação atual. Em áudio, isso revela **quais trechos** do sinal influenciam a decisão do modelo.

**Como obter (Hugging Face):** defina `output_attentions=True` no forward do modelo. Muitos encoders (Wav2Vec2/HuBERT/WavLM) expõem `attentions` por **camada × cabeça × tempo × tempo**.

```python
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, librosa, matplotlib.pyplot as plt

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Áudio padrão (mono, 16 kHz)
y, sr = librosa.load("voz.wav", sr=16000)
inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    # Lista de tensores: [num_camadas] cada um com (batch, heads, T, T)
    attentions = outputs.attentions

# Visualiza a 12ª camada, cabeça 0
att = attentions[-1][0, 0].cpu().numpy()  # (T, T)
plt.figure(figsize=(5,4))
plt.imshow(att, aspect='auto', origin='lower')
plt.title("Mapa de atenção – camada final, cabeça 0")
plt.xlabel("Passo de tempo (chave)")
plt.ylabel("Passo de tempo (query)")
plt.colorbar(label='peso')
plt.tight_layout(); plt.show()
```

**Agregações úteis:**
- **Por cabeça:** média ao longo de `heads` para ver um padrão global.
- **Por camada:** comparar camadas iniciais (padrões locais) vs finais (contexto longo).
- **Across-time:** somar por eixo para obter **importância temporal** agregada.

**Alinhamento com tempo real:** cada posição temporal do encoder corresponde a janelas/steps do extrator de *features* interno. Para leitura aproximada, mapeie `T` para segundos por `T * hop_seconds` (o *stride* do encoder; consulte a doc do modelo).

**Cuidados:**
- Atenção alta ≠ causalidade. Use atenção como **pista exploratória**, não prova.
- Camadas/cabeças diferentes capturam padrões distintos (prosódia, formantes, pausas).

---

### 7.2 Métodos XAI (Explainable AI)

#### 7.2.1 SHAP – impacto das *features*

**Ideia:** explicar a predição de um classificador downstream (SVM, RF, XGBoost, MLP) treinado sobre **embeddings**. O SHAP estima a contribuição de cada dimensão (ou *feature* agregada) para a saída.

**Exemplo com XGBoost (TreeExplainer):**
```python
import xgboost as xgb, shap
import numpy as np

# X_emb: (n_amostras, d) – embeddings; y: rótulos binários
clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)
clf.fit(X_emb, y)

explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_emb)

# Sumário global (quais dimensões/atributos mais importam)
shap.plots.beeswarm(shap_values, max_display=20)
```

**Dicas:**
- Para modelos não-arbóreos, use `shap.KernelExplainer` (mais lento) ou `shap.LinearExplainer`.
- Se você tem *features* híbridas (MFCC/CQCC + embeddings), una colunas e **nomeie-as** para relatórios.

---

#### 7.2.2 Grad-CAM adaptado para espectrogramas

**Ideia:** usar **Grad-CAM** em um **classificador de espectrogramas** (CNN simples) para destacar regiões tempo–frequência que sustentam a decisão. Para Transformers puros, o Grad-CAM direto é menos trivial; a prática comum é aplicar Grad-CAM no **modelo CNN** que consome **mel-espectrogramas** ou aplicar variantes em camadas convolucionais do *feature extractor*.

**Pipeline:** converter áudio → mel-espectrograma (como imagem) → CNN → Grad-CAM.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.feat = F.relu(self.conv2(x))  # mapa p/ Grad-CAM
        x = self.pool(self.feat).squeeze(-1).squeeze(-1)
        logit = self.fc(x)
        return logit

model = TinyCNN().eval()

# Suponha S_db ∈ R[n_mels, T] (mel-espectrograma em dB)
S_db = np.load("mel_db.npy")
X = torch.tensor(S_db[None, None, ...], dtype=torch.float32)

logit = model(X)
score = logit.squeeze()

# Grad-CAM: gradiente do score wrt feat-map
model.zero_grad(); score.backward(retain_graph=True)
grads = model.feat.grad if model.feat.requires_grad else None
if grads is None:
    model.feat.retain_grad()
    model.zero_grad(); score.backward()
    grads = model.feat.grad

weights = grads.mean(dim=(2,3), keepdim=True)  # média espacial
cam = (weights * model.feat).sum(dim=1, keepdim=True)  # (1,1,H,W)
cam = F.relu(cam).squeeze().detach().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

plt.figure(figsize=(8,3))
plt.imshow(S_db, aspect='auto', origin='lower', cmap='gray')
plt.imshow(cam, aspect='auto', origin='lower', alpha=0.45)
plt.title('Grad-CAM sobre mel-espectrograma')
plt.xlabel('Tempo'); plt.ylabel('Mel-bandas'); plt.tight_layout(); plt.show()
```

**Leitura do mapa:** regiões mais claras apontam **bandas/tempos** com maior influência na decisão (ex.: fricativas, harmônicos instáveis, artefatos de vocoder).

---

#### 7.2.3 Interpretação temporal de confiança (confiabilidade ao longo do tempo)

Para tarefas com **segmentos** (e.g., chunking de 2–4s), inspecione a **probabilidade prevista** em cada segmento para entender **onde** o modelo está mais/menos confiante.

```python
import numpy as np, matplotlib.pyplot as plt

# probs: vetor de probabilidades por segmento, timestamps: (início, fim) de cada segmento
probs = np.load("probs_por_segmento.npy")
centros = np.load("centros_temporais.npy")  # segundos

plt.figure(figsize=(8,3))
plt.plot(centros, probs, marker='o')
plt.axhline(0.5, ls='--')
plt.title('Confiança por tempo (probabilidade classe=1)')
plt.xlabel('Tempo (s)'); plt.ylabel('P(y=1)'); plt.tight_layout(); plt.show()
```

**Uso prático:**
- Relacione picos/vales de confiança com eventos acústicos (plosivas, pausas, ruídos). 
- Combine com **atenção temporal** para validar hipóteses do modelo.

---

### Boas práticas

- **Cross-check:** atenção e XAI devem ser cruzadas com sinais/figuras para evitar leituras equivocadas.
- **Calibração:** explique probabilidades **após** calibrar (Platt/Isotônico) para evitar falsas certezas.
- **Unidades:** deixe claro o mapeamento passo→tempo (stride, hop) ao rotular eixos.
- **Reprodutibilidade:** fixe sementes e documente versões de modelos e hiperparâmetros.

---

### Resumo

- **Atenção** mostra dependências temporais internas do Transformer.
- **SHAP** explica predições de classificadores treinados sobre **embeddings**.
- **Grad-CAM** (em CNNs de espectrograma) destaca regiões tempo–frequência decisivas.
- **Curvas de confiança no tempo** ajudam a localizar **onde** o modelo acerta/erra.

Combinadas, essas técnicas formam um **painel de interpretabilidade** robusto para auditoria, depuração de modelos e comunicação com especialistas do domínio.

---

## 8. Métricas e Avaliação

A avaliação de modelos de áudio com Transformers deve considerar **discriminação**, **calibração**, **agrupamento** e **tarefas específicas** como ASR. Cada categoria reflete um aspecto diferente da performance do sistema e, juntas, formam uma análise robusta e confiável.

---

### 8.1 Métricas de Discriminação

Avaliam o **poder de separação** do modelo entre classes distintas (ex.: fala real vs sintética, saudável vs patológica).

| Métrica | Descrição | Interpretação |
|----------|------------|---------------|
| **AUC (Area Under ROC Curve)** | Área sob a curva ROC; mede a capacidade de distinguir classes. | 1.0 = separação perfeita; 0.5 = aleatório. |
| **EER (Equal Error Rate)** | Ponto em que Falsos Positivos = Falsos Negativos. | Quanto menor, melhor (menor taxa de erro). |

#### Exemplo prático (Python)
```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

fpr, tpr, th = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
print(f"AUC: {auc:.4f}, EER: {eer:.4f}")
```

> **Dica:** AUC mostra a qualidade global da separação; EER foca em segurança/biometria, sendo muito usada em anti-spoofing.

---

### 8.2 Métricas de Calibração

Mesmo que um modelo acerte as classes, ele pode **superestimar ou subestimar** probabilidades. A calibração mede a **coerência entre probabilidade prevista e frequência observada**.

| Métrica | Descrição | Interpretação |
|----------|------------|---------------|
| **Brier Score** | Média dos erros quadráticos das probabilidades. | Menor = melhor (0 = perfeito). |
| **ECE (Expected Calibration Error)** | Diferença média entre probabilidade prevista e precisão real em intervalos. | Menor = melhor; ideal < 0.02. |

#### Exemplo prático
```python
from sklearn.calibration import calibration_curve, BrierScoreLoss
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('Probabilidade prevista')
plt.ylabel('Frequência observada')
plt.title('Curva de Calibração')
plt.tight_layout(); plt.show()
```

> **Interpretação:** linhas próximas da diagonal indicam boa calibração; curvas acima/abaixo mostram super/subestimação.

---

### 8.3 Métricas de Agrupamento

Usadas para avaliar embeddings sem rótulos explícitos (clustering de locutores, emoções, deepfakes, etc.).

| Métrica | Descrição | Interpretação |
|----------|------------|---------------|
| **Silhouette Score** | Mede coesão intra-cluster e separação inter-cluster. | −1 a 1; mais próximo de 1 é melhor. |
| **Davies–Bouldin Index** | Média da razão entre dispersão intra e distância inter-clusters. | Menor = melhor. |

#### Exemplo prático
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil = silhouette_score(X_emb, labels)
dbi = davies_bouldin_score(X_emb, labels)
print(f"Silhouette: {sil:.3f}, Davies–Bouldin: {dbi:.3f}")
```

> **Nota:** estas métricas ajudam a avaliar **qualidade estrutural** do espaço de representação antes de treinar classificadores.

---

### 8.4 Métricas para ASR (Reconhecimento de Fala)

Modelos como Whisper ou Wav2Vec2-ASR são avaliados por erro de transcrição. Duas métricas principais medem o alinhamento entre texto previsto e referência.

| Métrica | Descrição | Interpretação |
|----------|------------|---------------|
| **WER (Word Error Rate)** | Percentual de palavras erradas (inserções, deleções, substituições). | 0% = perfeito. |
| **CER (Character Error Rate)** | Percentual de erros em nível de caractere. | Mais sensível a erros pequenos. |

#### Exemplo prático (Hugging Face Evaluate)
```python
import evaluate
wer = evaluate.load('wer')
cer = evaluate.load('cer')

ref = ["ola mundo"]
pred = ["ola mudo"]
print("WER:", wer.compute(references=ref, predictions=pred))
print("CER:", cer.compute(references=ref, predictions=pred))
```

> **Dica:** use CER para idiomas foneticamente ricos (como português) e WER para análise semântica.

---

### 8.5 Boas práticas de avaliação

- **Cross-validation:** use *StratifiedGroupKFold* quando há dependência por locutor.  
- **Intervalos de confiança:** calcule desvios padrão das métricas (ex.: via *bootstrapping*).  
- **Calibração antes da interpretação:** avalie SHAP/atenção apenas em modelos bem calibrados.  
- **Métricas complementares:** combine discriminação + calibração para visão completa.

---

### 8.6 Resumo geral

| Tipo | Métrica | Interpretação |
|------|----------|---------------|
| **Discriminação** | AUC, EER | Separabilidade entre classes |
| **Calibração** | Brier, ECE | Coerência de probabilidades |
| **Agrupamento** | Silhouette, Davies–Bouldin | Qualidade estrutural de clusters |
| **ASR** | WER, CER | Erro de transcrição |

Cada métrica ilumina um aspecto da performance do sistema. Um modelo ideal apresenta **AUC alto**, **Brier/ECE baixos**, **Silhouette alto**, e **WER/CER reduzidos** — equilibrando precisão, confiança e interpretabilidade.
---

## 9. Tendências e Pesquisas Atuais

Os avanços recentes em aprendizado de máquina têm impulsionado o surgimento de **modelos foundation** que integram múltiplas modalidades — texto, imagem, fala e áudio — em arquiteturas unificadas. Este capítulo destaca as principais tendências que moldam o futuro da pesquisa e aplicação dos **Transformers em áudio**.

---

### 9.1 Modelos *Foundation* Multimodais

Os modelos *foundation* combinam **compreensão e geração** em diferentes domínios. Eles são pré-treinados em escalas massivas e posteriormente ajustados para tarefas específicas com poucos dados.

| Modelo | Organização | Características principais |
|---------|--------------|-----------------------------|
| **AudioGPT** | Microsoft | Integra fala, texto e música; interage com modelos de linguagem (GPT) para compreensão multimodal. |
| **SpeechLM** | Microsoft Research | Modelo de linguagem para fala, capaz de realizar *speech-to-speech* e *speech-to-text* em múltiplos idiomas. |
| **Gemini** | Google DeepMind | Integra visão, texto e fala em uma única arquitetura; foca em raciocínio multimodal. |

Esses modelos buscam **entendimento semântico unificado**: compreender o contexto auditivo de forma semelhante ao modo como os LLMs entendem linguagem textual.

#### Exemplos de avanços recentes:
- **Aprendizado contrastivo multimodal:** CLAP, AudioCLIP e Whisper-CLIP alinham representações entre som e texto.  
- **Modelos de geração de áudio:** MusicLM, AudioLDM e Jukebox produzem fala, música e efeitos sonoros de alta qualidade a partir de texto.  
- **Diálogo multimodal:** AudioGPT interpreta comandos de voz e responde com fala sintetizada e raciocínio contextual.

> **Impacto:** as fronteiras entre ASR, TTS, NLU e geração de conteúdo estão se dissolvendo — rumo a sistemas verdadeiramente *end-to-end* multimodais.

---

### 9.2 Fine-tuning Eficiente (LoRA, Adapters)

O aumento no tamanho dos modelos tornou essencial desenvolver **técnicas de adaptação leve**, que permitem personalizar modelos grandes sem re-treinar todos os parâmetros.

#### Principais abordagens:

| Técnica | Descrição | Vantagem |
|----------|------------|-----------|
| **LoRA (Low-Rank Adaptation)** | Insere matrizes de baixo posto nas camadas de atenção; apenas estas são treinadas. | Reduz o número de parâmetros treináveis em até 99%. |
| **Adapters** | Pequenos blocos adicionais inseridos entre camadas Transformer. | Permitem múltiplas tarefas em um único modelo base. |
| **Prompt-tuning / Prefix-tuning** | Introduz *prompts aprendíveis* que guiam a atenção do modelo. | Adaptação rápida sem alterar pesos originais. |

Essas estratégias tornam viável o uso de **modelos foundation** em dispositivos com recursos limitados, como aplicações embarcadas de voz e saúde.

> **Tendência:** o foco está migrando do *full fine-tuning* para abordagens modulares, eficientes e reutilizáveis — permitindo pesquisa aberta e colaboração.

---

### 9.3 *Continual Learning* e *Cross-domain Adaptation*

Outra linha emergente é o **aprendizado contínuo**, em que o modelo é atualizado de forma incremental conforme novas tarefas e domínios surgem — sem esquecer o conhecimento anterior (*catastrophic forgetting*).

#### Desafios atuais:
- **Domínios divergentes:** modelos treinados em fala leiga podem ter baixo desempenho em fala clínica, ruído urbano ou sotaques regionais.  
- **Aprendizado incremental:** adaptação de forma estável a novos idiomas, estilos vocais e ambientes.  
- **Avaliação contínua:** necessidade de benchmarks dinâmicos que reflitam o desempenho ao longo do tempo.

#### Abordagens promissoras:
- *Elastic Weight Consolidation (EWC)* e *Replay Memory* para preservar conhecimento antigo.  
- *Domain-Adaptive Fine-Tuning (DAFT)*: ajuste supervisionado em poucos dados do novo domínio.  
- *Cross-lingual transfer*: adaptação entre idiomas próximos usando embeddings compartilhados.

> **Aplicação direta:** modelos de voz usados em **telemedicina**, **educação inclusiva** e **verificação de identidade** se beneficiam do aprendizado contínuo e da adaptação contextual.

---

### 9.4 Aplicações em Segurança, Saúde e Acessibilidade

As fronteiras entre pesquisa e impacto social estão cada vez mais próximas. Modelos de áudio baseados em Transformers vêm transformando três grandes áreas:

#### 1. **Segurança Digital**
- Detecção e mitigação de **deepfakes** e fraudes por voz.  
- Verificação de locutores com calibração confiável (EER < 5%).  
- Monitoramento de autenticidade em aplicações forenses e bancárias.

#### 2. **Saúde Vocal e Cognitiva**
- Diagnóstico de doenças neurológicas (Parkinson, Alzheimer) e distúrbios vocais.  
- Avaliação de fadiga e esforço vocal em profissionais da voz.  
- Ferramentas para triagem remota em fonoaudiologia e telemedicina.

#### 3. **Acessibilidade e Inclusão**
- Sistemas TTS personalizados para pessoas com perda de voz.  
- Tradução e legendagem automática em tempo real.  
- Reconhecimento de fala robusto para múltiplos sotaques e dialetos.

> **Reflexão:** a pesquisa em áudio com Transformers transcende a tecnologia — impacta diretamente a **segurança, saúde e inclusão** humanas.

---

### 9.5 Perspectivas Futuras

O campo de áudio multimodal está em rápida expansão. Entre as direções mais promissoras estão:

- **Modelos unificados de percepção e geração** (SpeechLM, Whisper-v3, AudioGemini).  
- **Fusão de modalidades**: voz + vídeo + texto em pipelines de raciocínio.  
- **Explicabilidade integrada**: atenção auditiva interpretável nativamente.  
- **Sustentabilidade computacional**: pesquisa em modelos mais leves e acessíveis.

> O futuro dos Transformers em áudio está na convergência entre **eficiência, transparência e impacto social**, moldando uma nova geração de sistemas auditivos inteligentes.
---

## 10. Referências Selecionadas
1. Baevski et al. (2020). **wav2vec 2.0: A framework for self-supervised learning of speech representations.** *NeurIPS.*  
2. Hsu et al. (2021). **HuBERT: Self-supervised speech representation learning by masked prediction of hidden units.** *IEEE TASLP.*  
3. Chen et al. (2022). **WavLM: Large-scale self-supervised pre-training for full stack speech processing.** *IEEE J-STSP.*  
4. Radford et al. (2023). **Whisper: Robust speech recognition via large-scale weak supervision.** *arXiv:2212.04356.*  
5. Wu et al. (2023). **AudioMAE: Masked autoencoders are efficient learners for self-supervised audio pretraining.** *arXiv:2207.06405.*  
6. Elizalde et al. (2023). **CLAP: Learning audio-text joint embedding for retrieval and classification.** *IEEE ICASSP.*

