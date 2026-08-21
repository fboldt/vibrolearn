"""
cwru_loader.py

Carregador dos sinais CWRU a partir do diretório de arquivos .mat e do
arquivo de configuração (map) com metadados.

Saída principal
---------------
X            : ndarray, shape (n_segmentos, segment_length, 1)
fault_class  : ndarray, shape (n_segmentos,)
severity     : ndarray, shape (n_segmentos,)
load         : ndarray, shape (n_segmentos,)
metadata     : pandas.DataFrame com a origem de cada segmento

A estrutura de X é diretamente compatível com a classe WaveletPackage
fornecida no código de WPD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class CWRUData:
    """Contêiner para os segmentos carregados do CWRU."""
    X: np.ndarray
    fault_class: np.ndarray
    severity: np.ndarray
    load: np.ndarray
    metadata: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Resume o número de segmentos por classe, severidade e carga."""
        return (
            self.metadata
            .groupby(["condition", "severity", "load"], dropna=False)
            .size()
            .reset_index(name="n_segments")
            .sort_values(["condition", "severity", "load"])
            .reset_index(drop=True)
        )


def read_cwru_map(config_path: str | Path) -> pd.DataFrame:
    """
    Lê o arquivo de configuração no formato CSV mostrado no framework.

    O parser remove espaços excedentes dos nomes das colunas e dos campos
    textuais e converte valores 'None' para NaN.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    df = pd.read_csv(
        config_path,
        sep=",",
        skipinitialspace=True,
        na_values=["None", "none", "NULL", "null", ""],
        keep_default_na=True,
    )

    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "condition",
        "acquisition_file",
        "load",
        "severity",
        "sample_rate",
        "faulty_bearing",
        "DE",
        "FE",
        "BA",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "O arquivo de configuração não contém as colunas obrigatórias: "
            + ", ".join(sorted(missing))
        )

    for col in ["condition", "acquisition_file", "faulty_bearing", "DE", "FE", "BA"]:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    for col in ["load", "severity", "sample_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "prlz" in df.columns:
        df["prlz"] = pd.to_numeric(df["prlz"], errors="coerce")

    return df


def _normalize_sensor(sensor: str) -> str:
    sensor = sensor.upper().strip()
    allowed = {"DE", "FE", "BA"}
    if sensor not in allowed:
        raise ValueError(f"sensor deve ser um de {sorted(allowed)}; recebido: {sensor}")
    return sensor


def _segment_signal(
    signal: np.ndarray,
    segment_length: int,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segmenta um sinal 1D em janelas de comprimento fixo.

    Returns
    -------
    segments : ndarray, shape (n_segmentos, segment_length)
    starts   : ndarray com o índice inicial de cada segmento
    """
    signal = np.asarray(signal).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"O sinal precisa ser 1D após squeeze(); shape obtido: {signal.shape}")

    if segment_length <= 0:
        raise ValueError("segment_length deve ser positivo.")

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap deve satisfazer 0 <= overlap < 1.")

    step = int(round(segment_length * (1.0 - overlap)))
    step = max(step, 1)

    if signal.size < segment_length:
        return (
            np.empty((0, segment_length), dtype=np.float64),
            np.empty((0,), dtype=int),
        )

    starts = np.arange(
        0,
        signal.size - segment_length + 1,
        step,
        dtype=int,
    )

    segments = np.stack(
        [signal[start:start + segment_length] for start in starts],
        axis=0,
    ).astype(np.float64, copy=False)

    return segments, starts


def load_cwru(
    data_dir: str | Path,
    config_path: str | Path,
    *,
    sensor: str = "DE",
    segment_length: int = 2048,
    overlap: float = 0.0,
    sample_rate: Optional[int] = 48000,
    conditions: Optional[Sequence[str]] = ("Inner Race", "Ball", "Outer Race"),
    severities: Optional[Sequence[float]] = (0.007, 0.014, 0.021),
    loads: Optional[Sequence[int]] = (0, 1, 2, 3),
    faulty_bearing: Optional[str] = "Drive End",
    outer_race_position: Optional[int] = 6,
    only_existing_files: bool = True,
) -> CWRUData:
    """
    Carrega e segmenta os dados CWRU.

    Parâmetros
    ----------
    data_dir
        Diretório contendo os arquivos .mat, por exemplo "raw_data/cwru".

    config_path
        Caminho para o arquivo map/CSV mostrado no framework.

    sensor
        Canal a ser utilizado: "DE", "FE" ou "BA".

    segment_length
        Número de pontos em cada segmento.

    overlap
        Sobreposição entre janelas, no intervalo [0, 1).
        Para segmentos não sobrepostos, use 0.0.

    sample_rate
        Taxa de amostragem a selecionar. Para a análise dos 12 domínios,
        recomenda-se usar uma única taxa, por exemplo 48000.
        Use None para não filtrar.

    conditions
        Classes a carregar. Por padrão: Inner Race, Ball e Outer Race.

    severities
        Severidades de falha a carregar.

    loads
        Cargas a carregar.

    faulty_bearing
        Filtra o rolamento no qual a falha foi introduzida.
        Por padrão "Drive End". Use None para não filtrar.

    outer_race_position
        Para Outer Race, restringe a posição da falha (prlz), por padrão
        6 horas, evitando misturar posições diferentes. Use None para não
        aplicar essa restrição.

    only_existing_files
        Se True, ignora linhas do map cujos .mat não existem no diretório.
        Se False, lança FileNotFoundError.

    Retorno
    -------
    CWRUData
        X possui shape (n_segmentos, segment_length, 1), compatível com
        WaveletPackage.
    """
    data_dir = Path(data_dir)
    config_path = Path(config_path)
    sensor = _normalize_sensor(sensor)

    if not data_dir.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

    df = read_cwru_map(config_path)

    # --------------------------------------------------------
    # Filtros gerais
    # --------------------------------------------------------
    mask = np.ones(len(df), dtype=bool)

    if sample_rate is not None:
        mask &= df["sample_rate"].eq(sample_rate).to_numpy()

    if conditions is not None:
        mask &= df["condition"].isin(conditions).to_numpy()

    if severities is not None:
        sev = np.asarray(severities, dtype=float)
        mask &= df["severity"].apply(
            lambda x: np.any(np.isclose(x, sev)) if pd.notna(x) else False
        ).to_numpy()

    if loads is not None:
        mask &= df["load"].isin(loads).to_numpy()

    if faulty_bearing is not None:
        mask &= df["faulty_bearing"].eq(faulty_bearing).to_numpy()

    # O campo referente ao sensor precisa existir na linha.
    mask &= df[sensor].notna().to_numpy()

    selected = df.loc[mask].copy()

    # Evita misturar as posições 3, 6 e 12 horas de Outer Race.
    if outer_race_position is not None and "prlz" in selected.columns:
        is_outer = selected["condition"].eq("Outer Race")
        keep_outer = selected["prlz"].eq(outer_race_position)
        selected = selected.loc[(~is_outer) | keep_outer].copy()

    if selected.empty:
        raise ValueError(
            "Nenhuma aquisição atende aos filtros definidos. "
            "Verifique sample_rate, classes, severidades, cargas, "
            "faulty_bearing e sensor."
        )

    # --------------------------------------------------------
    # Carregamento dos arquivos e segmentação
    # --------------------------------------------------------
    X_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for _, row in selected.iterrows():
        mat_path = data_dir / str(row["acquisition_file"]).strip()

        if not mat_path.exists():
            if only_existing_files:
                print(f"[AVISO] Arquivo ausente; ignorado: {mat_path}")
                continue
            raise FileNotFoundError(f"Arquivo .mat não encontrado: {mat_path}")

        mat = loadmat(mat_path)

        variable_name = str(row[sensor]).strip()

        if variable_name not in mat:
            available = sorted(k for k in mat.keys() if not k.startswith("__"))
            raise KeyError(
                f"Variável '{variable_name}' não encontrada em {mat_path.name}. "
                f"Variáveis disponíveis: {available}"
            )

        signal = np.asarray(mat[variable_name]).squeeze()

        segments, starts = _segment_signal(
            signal,
            segment_length=segment_length,
            overlap=overlap,
        )

        if len(segments) == 0:
            print(
                f"[AVISO] {mat_path.name}: sinal menor que "
                f"segment_length={segment_length}; ignorado."
            )
            continue

        # WaveletPackage espera:
        # (n_amostras, comprimento_segmento, n_canais)
        segments = segments[:, :, np.newaxis]
        X_parts.append(segments)

        file_meta = pd.DataFrame({
            "condition": row["condition"],
            "severity": float(row["severity"]),
            "load": int(row["load"]),
            "sample_rate": int(row["sample_rate"]),
            "faulty_bearing": row["faulty_bearing"],
            "sensor": sensor,
            "mat_file": mat_path.name,
            "mat_variable": variable_name,
            "segment_index": np.arange(len(segments), dtype=int),
            "start_sample": starts,
            "end_sample": starts + segment_length,
        })

        if "prlz" in row.index:
            file_meta["prlz"] = row["prlz"]

        meta_parts.append(file_meta)

    if not X_parts:
        raise RuntimeError(
            "Nenhum segmento foi carregado. Verifique se os arquivos .mat "
            "selecionados realmente existem no diretório informado."
        )

    X = np.concatenate(X_parts, axis=0)
    metadata = pd.concat(meta_parts, ignore_index=True)

    # Arrays esperados pelo código de MMD fornecido anteriormente.
    fault_class = metadata["condition"].to_numpy(dtype=str)
    severity = metadata["severity"].to_numpy(dtype=float)
    load = metadata["load"].to_numpy(dtype=int)

    return CWRUData(
        X=X,
        fault_class=fault_class,
        severity=severity,
        load=load,
        metadata=metadata,
    )


def validate_12_domains(
    data: CWRUData,
    *,
    classes: Sequence[str] = ("Inner Race", "Ball", "Outer Race"),
    severities: Sequence[float] = (0.007, 0.014, 0.021),
    loads: Sequence[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """
    Verifica quais combinações classe x severidade x carga estão presentes.
    """
    rows = []

    for cls in classes:
        for sev in severities:
            for ld in loads:
                mask = (
                    (data.fault_class == cls)
                    & np.isclose(data.severity, sev)
                    & (data.load == ld)
                )
                rows.append({
                    "condition": cls,
                    "severity": sev,
                    "load": ld,
                    "n_segments": int(mask.sum()),
                    "available": bool(mask.any()),
                })

    return pd.DataFrame(rows)