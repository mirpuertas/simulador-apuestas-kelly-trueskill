import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kelly import simulate, performance_stats, compute_result

class KellyBettingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kelly Betting Strategy Simulator")
        self.root.geometry("1200x800")
        
        # Crear el marco principal
        self.create_input_frame()
        self.create_plot_frame()
        
        # Inicializar las variables
        self.csv_path = tk.StringVar(value="Trabajo-Final/Final-Apuestas/inferencia/predicciones_wta.csv")
        self.bankroll = tk.DoubleVar(value=1000.0)
        self.kelly_fraction = tk.DoubleVar(value=0.01)
        self.risk_cap = tk.DoubleVar(value=0.03)
        self.min_edge = tk.DoubleVar(value=0.01)
        self.beta = tk.DoubleVar(value=1.0)
        self.sigma_thresh = tk.DoubleVar(value=2.0)
        
        # Graficar las selecciones
        self.graph_type = tk.StringVar(value="both")
        
        # Crear input widgets
        self.create_input_widgets()
        
    def create_input_frame(self):
        self.input_frame = ttk.LabelFrame(self.root, text="Simulation Parameters", padding="10")
        self.input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Creamos un Notebook con dos pestañas
        self.tab_control = ttk.Notebook(self.input_frame)
        self.tab_basic   = ttk.Frame(self.tab_control)
        self.tab_advanced = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_basic,   text="Básicos")
        self.tab_control.add(self.tab_advanced, text="Avanzados")
        self.tab_control.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.input_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))

        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, pady=(0,10))
        self.run_button = ttk.Button(button_frame, text="Simular", command=self.run_simulation)
        self.run_button.pack(pady=5)

        results_frame = ttk.LabelFrame(bottom_frame, text="Resultados de la Simulación", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_frame, wrap="word", height=15, width=40)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        self.results_text.config(font=("Consolas", 10))
        self.results_text.config(state="disabled")
        
    def create_plot_frame(self):
        self.plot_frame = ttk.LabelFrame(self.root, text="Resultados", padding="10")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_input_widgets(self):
 
        ttk.Label(self.tab_basic, text="CSV File:").pack(anchor=tk.W)
        file_frame = ttk.Frame(self.tab_basic)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.csv_path, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
  
        self.create_numeric_input("Bankroll incial:", self.bankroll, self.tab_basic)
        self.create_numeric_input("Kelly Fraction:", self.kelly_fraction, self.tab_basic)
        self.create_numeric_input("Risk Cap:", self.risk_cap, self.tab_basic)

        ttk.Label(self.tab_basic, text="Stake Mode:").pack(anchor=tk.W)
        self.stake_mode = ttk.Combobox(self.tab_basic, values=["kelly", "frac", "flat", "edge"], state="readonly")
        self.stake_mode.set("kelly")
        self.stake_mode.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.tab_basic, text="Bet Mode:").pack(anchor=tk.W)
        self.bet_mode = ttk.Combobox(self.tab_basic, values=["winner", "loser", "best", "both"], state="readonly")
        self.bet_mode.set("loser")
        self.bet_mode.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.tab_basic, text="Tipo de Gráfico:").pack(anchor=tk.W)
        graph_frame = ttk.Frame(self.tab_basic)
        graph_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(graph_frame, text="Linear", variable=self.graph_type, value="linear").pack(side=tk.LEFT)
        ttk.Radiobutton(graph_frame, text="Log", variable=self.graph_type, value="log").pack(side=tk.LEFT)
        ttk.Radiobutton(graph_frame, text="Both", variable=self.graph_type, value="both").pack(side=tk.LEFT)
        
        self.adjust_vig = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.tab_basic, text="Ajustar Vig", variable=self.adjust_vig).pack(anchor=tk.W, pady=5)
        
        ttk.Label(self.tab_advanced, text="Parámetros Avanzados", font=("Consolas", 10, "bold")).pack(anchor=tk.W, pady=(0,10))
        self.create_numeric_input("Edge mínimo:", self.min_edge, self.tab_advanced)
        self.create_numeric_input("Beta:", self.beta, self.tab_advanced)
        self.create_numeric_input("Sigma (Umbral):", self.sigma_thresh, self.tab_advanced)
        
    def create_numeric_input(self, label, variable, parent):
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=variable).pack(fill=tk.X, pady=5)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path.set(filename)
            
    def run_simulation(self):
        try:
            df = pd.read_csv(self.csv_path.get(), parse_dates=["date"])
            df = df.sort_values("date")
            df = df.dropna(subset=["m_winner", "m_loser", "s_winner", "s_loser", "b_winner", "b_loser"])
            df[["b_winner", "b_loser"]] = df[["b_winner", "b_loser"]].apply(pd.to_numeric, errors="coerce")
            
            df["result"] = df.apply(compute_result, axis=1)
            accuracy = (df["result"].sum() / len(df)) * 100
            
            log = simulate(
                df,
                self.bankroll.get(),
                self.kelly_fraction.get(),
                self.risk_cap.get(),
                self.stake_mode.get(),
                self.bet_mode.get(),
                self.adjust_vig.get(),
                self.min_edge.get(),
                sigma_thresh=self.sigma_thresh.get(),
                beta=self.beta.get()
            )

            log["date"] = pd.to_datetime(log["date"])
            log = log.sort_values("date")
            # print(log["date"].dtype)
            # print("Fechas en LOG:", log["date"].min(), "→", log["date"].max())
            

            stats = performance_stats(log)
            self.display_results(stats, accuracy, log.attrs["placed"], log.attrs["skipped"])
            

            self.update_plots(log)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def display_results(self, stats, accuracy, placed_bets, skipped_bets):
        self.results_text.config(state="normal") 
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "Estadísticas de Performance:\n", "header")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        for k, v in stats.items():
            self.results_text.insert(tk.END, f"{k:12}: {v:.4f}\n")

        self.results_text.insert(tk.END, "\nAccuracy de la Predicción:\n", "header")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Accuracy del modelo: {accuracy:.2f}%\n\n")
        
        self.results_text.insert(tk.END, "\nInformación de Apuestas:\n", "header")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Total de apuestas realizadas: {placed_bets}\n")
        self.results_text.insert(tk.END, f"Total de apuestas salteadas: {skipped_bets}\n")


        self.results_text.tag_configure("header", font=("Consolas", 10, "bold"))
        
        self.results_text.config(state="disabled") 
        
    def update_plots(self, log):
        # Limpiar el panel
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        mode = self.graph_type.get()

        if mode == "linear":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(log.date, log.bankroll, color="tab:blue")
            ax.set_title("Bankroll (Linear Scale)")
            ax.set_ylabel("Bankroll")
            ax.grid(True)
            
        elif mode == "log":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(log.date, log.bankroll, color="tab:green")
            ax.set_yscale("log")
            ax.set_title("Bankroll (Logarithmic Scale)")
            ax.set_ylabel("Bankroll (log)")
            ax.set_xlabel("Date")
            ax.grid(True, which="both", ls="--", alpha=0.5)
        elif mode == "both":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            ax1.plot(log.date, log.bankroll, color="tab:blue")
            ax1.set_title("Bankroll (Linear Scale)")
            ax1.set_ylabel("Bankroll")

            ax2.plot(log.date, log.bankroll, color="tab:green")
            ax2.set_yscale("log")
            ax2.set_title("Bankroll (Logarithmic Scale)")
            ax2.set_ylabel("Bankroll (log)")
            ax2.set_xlabel("Date")
        else:
            messagebox.showerror("Error", f"Unknown graph type: {mode}")
            return

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = KellyBettingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 