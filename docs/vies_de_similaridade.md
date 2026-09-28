# Viés de Similaridade

Anyssa Satimi Komati Andrade

## 1. O problema central: A ilusão da Acurácia perfeita

Na literatura de diagnóstico de falhas mecânicas (usando sinais de vibração de rolamentos), é comum encontrar taxas de acerto próximas a 100%. Porém, esses resultados costumam ser **artificialmente inflados** devido a falhas na validação experimental.
O principal culpado por essa distorção é o **viés de similaridade** (*similarity bias*).

## 2. O que é o Viés de similaridade?

Para treinar algoritmos de inteligência artificial com sinais contínuos de vibração (capturados de máquinas), o arquivo é dividido em milhares de pequenos trechos chamados de **chunks**.

- **O erro comum:** O viés de similaridade acontece quando trechos vizinhos extraídos do **mesmo arquivo de sinal** são distribuídos simultaneamente para o conjunto de **treino**  e para o conjunto de **teste**
- **Por que isso é um problema?** Como esses  trechos são quase idênticos, o modelo não aprende de fato a diagnosticar uma falha nova; ele apenas **memoriza o formato daquele arquivo específico**.
- **A consequência prática:** Quando o modelo é testado em dados de uma máquina real ou sob novas condições de operação, seu desempenho despenca drasticamente.

## 3. A solução Proposta pelos Autores 

Para eliminar esse viés e obter resultados confiáveis, o artigo propõe um novo framework metodológico:

1. **Validação Cruzada Aninhada (*nested CV*):** Utiliza dois laços. O laço interno ajusta as configurações do modelo (hiperparâmetros) usando apenas dados de treino; o laço externo avalia o desempenho final em um conjunto de teste **totalmente isolado**.
2. **Divisão Rígida por Condição Operacional:** Os dados são separados por cenários reais (como diferentes cargas no motor ou severidade de falha), garantindo que treino e teste nunca compartilhem pedaços do mesmo sinal.
3. **Testes Estatísticos Rigorosos:** Aplicação de testes (como Nadeau-Bengio) para provar se a diferença de desempenho entre os modelos analisados (KNN, SVM, Random Forest, Redes Neurais) é estatisticamente real e não apenas fruto do acaso.

## 4. O Impacto Prático (Estudo de Caso CWRU)

Os autores testaram diferentes classificadores no conhecido banco de dados de rolamentos da *Case Western Reserve University (CWRU)* sob dificuldades crescentes:

| **Cenário de Avaliação** | **O que o modelo enfrenta** | **Acurácia Média** |
|---|---|---|
| **1. Memorização**  | Divisão ingênua tradicional (com viés de similaridade) | **Extremamente alta** (falsamente otimista)|
| **2. Generalização por Carga** | Testa em uma carga de motor nunca vista no treino | **Alta** (ainda otimista)
| **3. Cenário Realista** | Sem viés de similaridade e com variações severas de operação | **Queda significativa** |

**Conclusão principal:** Quando o viés de similaridade é removido e o algoritmo é forçado a lidar com cenários reais, a acurácia cai pela metade, provando que muitos estudos anteriores apresentavam uma ilusão de perfeição.

## 5. Glossário Didático 

- **Viés de Similaridade (*Similarity Bias*):** Otimismo artificial gerado quando trechos quase idênticos de um mesmo sinal de dados aparecem tanto no treino quanto no teste.
- **Validação Cruzada Aninhada:** Método de teste dividido em duas camadas (interna e externa) para impedir que os dados de teste influenciem o treinamento do modelo.
- **Hiperparâmetros:** Ajustes prévios de configuração de um algoritmo de IA (ex: número de árvores em uma *Random Forest*) definidos antes do aprendizado real começar.
- **Overfitting de Hiperparâmetros:** O erro de ajustar repetidamente o modelo olhando  para o resultado do teste até ele parecer bom "por sorte" naquele cenário específico.
- **Sinais de vibração:** Dados físicos coletados  de máquinas para monitorar a saúde mecânica e identificar defeitos em peças como rolamentos.

