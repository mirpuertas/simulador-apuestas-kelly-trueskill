'''
Simulador de apuestas de Kelly sobre predicciones TrueSkill-Through-Time
---------------------------------------------------------------
Descripción:
  - Calcula probabilidades de victoria con TrueSkill (prob_win).
  - Aplica el criterio de Kelly (kelly_fraction) con controles prácticos.
  - Simula staking, genera métricas de performance y permite CLI de uso.
Alumno: Miguel Rodríguez Puertas
Institución: Universidad Nacional de General San Martín
Fecha: 2025-06-08
'''

from __future__ import annotations
import argparse
import math
import warnings
from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def prob_win(mu_w: float, mu_l: float, sig_w: float, sig_l: float, beta: float = 1.0) -> float:
    """
    Calcula P(win) usando el modelo TrueSkill:
    p = Φ((μ_w - μ_l) / sqrt(2β² + σ_w² + σ_l²))

    Parámetros
    ----------
    mu_w : float
        Media posterior del jugador favorito.
    mu_l : float
        Media posterior del jugador underdog.
    sig_w : float
        Desvío estándar posterior del favorito.
    sig_l : float
        Desvío estándar posterior del underdog.
    beta : float, opcional
        Parámetro de ruido del rendimiento (default=1.0).

    Devuelve
    -------
    float
        Probabilidad de victoria del favorito.
    """
    # Varianza compuesta: rendimiento vs. habilidad + incertidumbres
    var = 2 * beta ** 2 + sig_w ** 2 + sig_l ** 2
    return float(norm.cdf((mu_w - mu_l) / math.sqrt(var)))


def kelly_fraction(p: float, q: float) -> float:
    """
    Fracción óptima de Kelly para cuota q y probabilidad p.

    Fórmula clásica:
      f* = (p·b - (1-p)) / b,  donde b = q - 1

    Parámetros
    ----------
    p : float
        Probabilidad de éxito (0 < p < 1).
    q : float
        Cuota decimal (> 1).

    Devuelve
    -------
    float
        Fracción óptima (>= 0), 0 si edge <= 0.
    """
    # Validación de rangos
    if q <= 1.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    b = q - 1.0 # payout neto
    return max((p * b - (1.0 - p)) / b, 0.0)

def compute_result(row: pd.Series) -> int:
    """
    Determina el resultado real del partido: 1 si ganó el 'winner', 0 si no.

    row debe tener 'winner', 'loser', 'm_winner', 'm_loser'.
    """
    # Identificar favorito según media de habilidad
    fav = row.winner if row.m_winner >= row.m_loser else row.loser
    return 1 if fav == row.winner else 0


def simulate(df: pd.DataFrame,
             bankroll0: float,
             kelly_frac: float,
             risk_cap: float,
             stake_mode: Literal['kelly','frac','flat','edge'],
             bet_mode: Literal['winner','loser','best','both'],
             adjust_vig: bool,
             min_edge: float = 0.0,
             sigma_thresh: float = 2.0,
             beta: float = 1.0) -> pd.DataFrame:
    """
    Simula las apuestas sobre las predicciones de TrueSkill.

    Parámetros
    ----------
    df : DataFrame
        Columnas requeridas: ['date','m_winner','m_loser','s_winner','s_loser','b_winner','b_loser'].
    bankroll0 : float
        Bankroll inicial.
    kelly_frac : float
        Fracción (λ) para Kelly fraccional.
    risk_cap : float
        % máximo del bankroll arriesgado por apuesta.
    stake_mode : str
        'kelly','frac','flat' o 'edge'.
    bet_mode : str
        'winner','loser','best' o 'both'.
    adjust_vig : bool
        Si True, ajusta cuotas para eliminar overround.
    min_edge : float
        Umbral mínimo de EV para apostar.
    sigma_thresh : float
        Umbral de desviación para micro-stake.
    beta : float
        Volatilidad del modelo TrueSkill.

    Retorna
    -------
    DataFrame
        Registros de bankroll y estadísticas por fecha.
    """
    bank = bankroll0
    log: list[dict] = []            # Acumula registros de cada fecha
    n_skipped, n_bets = 0, 0        # Contadores de apuestas

    for _, row in df.iterrows():
        qw, ql = row.b_winner, row.b_loser

         # Ajuste de vig: normaliza cuotas para eliminar margen de la casa
        if adjust_vig:
            over = 1.0 / qw + 1.0 / ql
            qw, ql = qw / over, ql / over

        # Asignar favorito y underdog según media de skill
        if row.m_winner >= row.m_loser:
            mu_fav, sig_fav = float(row.m_winner), float(row.s_winner)
            mu_und, sig_und = float(row.m_loser), float(row.s_loser)
            odd_fav, odd_und = qw, ql
            result_fav = row.result # 1 si gana favorito, 0 si no
        else:
            mu_fav, sig_fav = float(row.m_loser), float(row.s_loser)
            mu_und, sig_und = float(row.m_winner), float(row.s_winner)
            odd_fav, odd_und = ql, qw
            result_fav = row.result

        # Probabilidad de victoria del favorito
        p_fav = prob_win(mu_fav, mu_und, sig_fav, sig_und, beta=beta)

        # Fracciones de Kelly sin escala
        f_fav = kelly_fraction(p_fav, odd_fav)
        f_und = kelly_fraction(1.0 - p_fav, odd_und)

        # Calcular edge para filtros
        edge_fav = p_fav * (odd_fav - 1.0) - (1.0 - p_fav)
        edge_und = (1.0 - p_fav) * (odd_und - 1.0) - p_fav

        # Control de incertidumbre: micro-stake si σ alta
        if max(sig_fav, sig_und) > sigma_thresh:
            tiny_frac = 0.001
            stake_fav = bank * tiny_frac if f_fav > 0 else 0.0
            stake_und = bank * tiny_frac if f_und > 0 else 0.0
        
        # Evitar apuestas de bajo edge
        elif max(edge_fav, edge_und) < min_edge:
            n_skipped += 1
            log.append({"date": row.date, "bankroll": bank, "log_bankroll": math.log(bank + 1e-9),
                        "stake_fav": 0.0, "stake_und": 0.0, "kelly_fav": f_fav, "kelly_und": f_und,
                        "edge_fav": edge_fav, "edge_und": edge_und, "p_win": p_fav,
                        "gain": 0.0, "correct": 0, "note": "skip_edge"})
            continue

        # Selección de sides según bet_mode
        else:
            if bet_mode == "winner": f_und = 0.0
            elif bet_mode == "loser": f_fav = 0.0
            elif bet_mode == "best":
                if f_fav >= f_und: f_und = 0.0
                else: f_fav = 0.0

            # Función interna para calcular stake según modo
            def _stake(frac):
                if stake_mode == "kelly": return bank * frac * kelly_frac
                if stake_mode == "frac": return bank * kelly_frac
                if stake_mode == "edge": return bank * min(frac, 1.0) * kelly_frac
                if stake_mode == "flat": return bankroll0 * kelly_frac
                raise ValueError("stake_mode inválido")

            # Calcular stakes y aplicar cap de riesgo
            stake_fav = min(_stake(f_fav), bank * risk_cap)
            stake_und = min(_stake(f_und), bank * risk_cap)

         # Si no hay stake en ninguno, registrar y seguir
        if stake_fav == 0 and stake_und == 0:
            log.append({"date": row.date, "bankroll": bank, "log_bankroll": math.log(bank + 1e-9),
                        "stake_fav": 0.0, "stake_und": 0.0, "kelly_fav": f_fav, "kelly_und": f_und,
                        "edge_fav": edge_fav, "edge_und": edge_und, "p_win": p_fav,
                        "gain": 0.0, "correct": 0, "note": "zero_stake"})
            continue
        # Calcular ganancia para favorito y underdog
        gain_fav = stake_fav * ((odd_fav - 1) * result_fav - (1 - result_fav))
        result_und = 1 - result_fav
        gain_und = stake_und * ((odd_und - 1) * result_und - (1 - result_und))
        gain = gain_fav + gain_und

        bank += gain # Actualizar bankroll
        correct = 1 if gain > 0 else 0
        n_bets += 1

        # Registrar en log
        log.append({"date": row.date, "bankroll": bank, "log_bankroll": math.log(bank + 1e-9),
                    "stake_fav": stake_fav, "stake_und": stake_und, "kelly_fav": f_fav, "kelly_und": f_und,
                    "edge_fav": edge_fav, "edge_und": edge_und, "p_win": p_fav,
                    "gain": gain, "correct": correct, "note": ""})

    # Construir DataFrame final con métricas
    out = pd.DataFrame(log)
    out.attrs["skipped"] = n_skipped
    out.attrs["placed"] = n_bets
    out["daily_log_return"] = out["log_bankroll"].diff().fillna(0.0)
    print(f"Total de apuestas: {n_bets}, saltadas: {n_skipped}")
    return out


def performance_stats(log: pd.DataFrame) -> dict[str, float]:
    """
    Calcula métricas de performance a partir del log de simulación.

    Parámetros
    ----------
    log : DataFrame
        DataFrame de salida de simulate con 'bankroll' y 'daily_log_return'.

    Devuelve
    -------
    dict[str, float]
        Métricas: Final, CAGR, Sharpe, MaxDD, ROI.
    """

    log = log.dropna(subset=["bankroll"])
    if log.empty:
        return {k: float("nan") for k in ["Final", "CAGR", "Sharpe", "MaxDD", "ROI"]}
    bk0, bkf = log.bankroll.iloc[0], log.bankroll.iloc[-1]

    # Años transcurridos
    years = (log.date.iloc[-1] - log.date.iloc[0]).days / 365.25
    cagr = (bkf / bk0) ** (1 / years) - 1 if years > 0 else float("nan")

    # Sharpe anualizado (√252 días)
    sharpe = log.daily_log_return.mean() / log.daily_log_return.std() * math.sqrt(252)

    # Max Drawdown
    cummax = log.bankroll.cummax()
    max_dd = ((cummax - log.bankroll) / cummax).max()

    # ROI total
    total_stake = log.get("stake_fav", 0).fillna(0) + log.get("stake_und", 0).fillna(0)
    total_gain = log.get("gain", 0).fillna(0)
    roi = total_gain.sum() / total_stake.sum() if total_stake.sum() > 0 else float("nan")
    return {"Final": bkf, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "ROI": roi}


def parse_cli() -> argparse.Namespace:
    """
    Parser de argumentos CLI para el simulador.
    """
    pa = argparse.ArgumentParser()
    pa.add_argument("csv", nargs="?", default="Trabajo-Final/Final-Apuestas/inferencia/predicciones_wta.csv", type=Path)
    pa.add_argument("--bank", type=float, default=1_000.0)
    pa.add_argument("--lambda", dest="kelly_frac", type=float, default=0.01)
    pa.add_argument("--cap", dest="risk_cap", type=float, default=0.03)
    pa.add_argument("--stake", dest="stake_mode", choices=["kelly", "frac", "flat", "edge"], default="kelly")
    pa.add_argument("--mode", dest="bet_mode", choices=["winner", "loser", "best", "both"], default="loser")
    pa.add_argument("--beta", type=float, default=1.0)
    pa.add_argument("--min-edge", dest="min_edge", type=float, default=0.01)
    pa.add_argument("--no-vig", action="store_true", default=True)
    pa.add_argument("--plot", action="store_true", default=True)
    pa.add_argument("--out", dest="out_path", type=Path)
    return pa.parse_args()


def main() -> None:
    """
    Función principal: parsea CLI, carga CSV, simula y muestra métricas.
    """
    args = parse_cli()
    # Carga datos y filtra filas incompletas
    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df.dropna(subset=["m_winner", "m_loser", "s_winner", "s_loser", "b_winner", "b_loser"])
    # Asegurar tipos numéricos en cuotas
    df[["b_winner", "b_loser"]] = df[["b_winner", "b_loser"]].apply(pd.to_numeric, errors="coerce")
    # Resultado real para comparar
    df["result"] = df.apply(compute_result, axis=1)
    print((df["result"] == 0).sum())

    # Ejecutar simulación
    log = simulate(df, args.bank, args.kelly_frac, args.risk_cap, args.stake_mode, args.bet_mode,
                   not args.no_vig, args.min_edge, sigma_thresh=2.0, beta=args.beta)
    
    # Calcular y mostrar estadísticas
    stats = performance_stats(log)
    print("\nResultados:")
    for k, v in stats.items():
        print(f"{k:12}: {v:.4f}")
    if args.out_path:
        log.to_csv(args.out_path, index=False)

    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Eje lineal
        ax1.plot(log.date, log.bankroll, color="tab:blue")
        ax1.set_title("Bankroll (escala lineal)")
        ax1.set_ylabel("Bankroll")
        ax1.grid(True)

        # Eje logarítmico
        ax2.plot(log.date, log.bankroll, color="tab:green")
        ax2.set_yscale("log")
        ax2.set_title("Bankroll (escala logarítmica)")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("Bankroll (log)")
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    # Graficar el overround!!!

    # df["overround"] = 1 / df["b_winner"] + 1 / df["b_loser"]
    # plt.plot(df["date"], df["overround"])
    # plt.axhline(1.0, color="gray", linestyle="--", label="Fair Odds")
    # plt.title("Overround a lo largo del tiempo")
    # plt.ylabel("1/q_w + 1/q_l")
    # plt.grid(True)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()