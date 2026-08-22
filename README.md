# VibroLearn

**VibroLearn** é um framework em Python para execução de experimentos de aprendizado de máquina aplicados ao diagnóstico de falhas em rolamentos a partir de sinais de vibração.

O framework foi desenvolvido com foco na separação entre **métodos de diagnóstico** e **protocolos experimentais**, permitindo avaliar diferentes modelos sob diferentes estratégias de treinamento, validação e teste.

A metodologia utilizada no desenvolvimento do framework e dos experimentos está descrita na dissertação:

> **Um Novo Protocolo Experimental para Avaliação da Generalização entre Domínios no Diagnóstico de Falhas em Rolamentos**
> [Acessar dissertação](URL_DA_DISSERTACAO)

---

## Métodos disponíveis

Atualmente, o VibroLearn possui suporte aos seguintes métodos:

* **WPD + SCED + Random Forest**
* **DINOv2**
* **CNN-LSTM**

O método WPD + SCED + RF combina extração de características por *Wavelet Packet Decomposition*, Seleção de Características Estáveis entre Domínios e classificação por Random Forest.

Os métodos são implementados no diretório:

```text
estimators/
```

e registrados em:

```text
estimators/registry.py
```

---

## Protocolos experimentais

Os protocolos são definidos por arquivos de configuração JSON e determinam como os dados são organizados para treinamento, validação e teste.

Entre os protocolos utilizados no projeto estão:

* Sehri-Khalilian;
* Sehri-Inspired;
* protocolo proposto na dissertação.

A definição do protocolo é independente do método utilizado.

---

## Estrutura

```text
vibrolearn/
│
├── main.py
├── requirements.txt
│
├── dataset/
├── estimators/
├── experiment/
├── feature/
├── preprocessing/
└── results/
```

Principais diretórios:

* `dataset/`: carregamento e organização das bases de dados;
* `estimators/`: implementação dos métodos de diagnóstico;
* `experiment/`: execução e avaliação dos experimentos;
* `feature/`: extração e seleção de características;
* `preprocessing/`: operações de pré-processamento;
* `results/`: resultados gerados pelos experimentos.

---

## Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd vibrolearn
```

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Execução

A execução segue a estrutura:

```bash
python main.py -m <metodo> -e <protocolo.json>
```

### WPD + SCED + RF

```bash
python main.py -m wpd_sced_rf -e <protocolo.json>
```

### DINOv2

```bash
python main.py -m dinov2 -e <protocolo.json>
```

### CNN-LSTM

```bash
python main.py -m cnn_lstm -e <protocolo.json>
```

O método e o protocolo são tratados de forma independente pelo framework.

---

## Adicionando novos métodos

Novos métodos devem implementar a interface utilizada pelo executor:

```python
class NewMethod:

    name = "new_method"

    def configurations(self):
        yield {}

    def build(self, configuration=None):
        return pipeline

    def metadata(self, configuration=None):
        return {"method": self.name}
```

Após a implementação, o método deve ser registrado em:

```text
estimators/registry.py
```

Dessa forma, novos métodos podem ser adicionados sem modificar a `main.py` ou a lógica geral de execução.

---

## Resultados

Os resultados são armazenados em arquivos JSON contendo, entre outras informações:

* método utilizado;
* protocolo experimental;
* repetição;
* hiperparâmetros relevantes;
* acurácia;
* F1-macro;
* matriz de confusão;
* tempos de execução.

---

## Licença

Consulte o arquivo:

```text
LICENSE
```
