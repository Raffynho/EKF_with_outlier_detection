import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importa le classi e funzioni dai file forniti
from unicycle_estimation import SimulationWithEstimation, FPS
from plot import plot_outlier_performance
from c_plot import analyze_and_plot_confusion

class AutomatedSimulation(SimulationWithEstimation):
    def __init__(self, max_seconds=20, imu_prob=0.01, cam_prob=0.01):
        super().__init__()
        self.max_frames = max_seconds * FPS
        
        # Sovrascrive dinamicamente le probabilità di outlier
        self.imu_outlier_prob = imu_prob
        self.cam_outlier_prob = cam_prob
        
    def update(self, frame):
        """Sovrascrive il metodo update per far muovere l'uniciclo a spirale."""
        t = self.frame_i / FPS
        
        # Percorso a spirale:
        # Velocità lineare crescente e velocità angolare costante
        self.uni.v_cmd = 0.5 * t 
        self.uni.theta_cmd = 1.5 * t
        
        # Esegui la logica originale (EKF, Outliers, Log, Plotting)
        res = super().update(frame)
        
        # Condizione di terminazione automatica
        if self.frame_i >= self.max_frames:
            print(f"\n[AutoSim] Raggiunti i {self.max_frames} frame. Termino la simulazione...")
            # Ferma l'animazione e salva i log in CSV
            self.on_close(None)
            # Chiude la finestra del grafico
            plt.close(self.fig)
            
        return res

def run_automated_analysis(imu_prob=0.01, cam_prob=0.01):
    # 1. Avvia la simulazione automatizzata
    print(f"=== AVVIO SIMULAZIONE (Prob. Outlier -> IMU: {imu_prob*100}%, Cam: {cam_prob*100}%) ===")
    sim = AutomatedSimulation(max_seconds=20, imu_prob=imu_prob, cam_prob=cam_prob)
    sim.run()
    
    # 2. Avvia l'analisi dei risultati temporali
    print("\n=== AVVIO ANALISI DEI DATI ===")
    plot_outlier_performance("dati_sensori.csv")
    
    # 3. Avvia l'analisi delle matrici di confusione
    print("\nGenerazione delle matrici di confusione...")
    df = pd.read_csv("dati_sensori.csv")
    
    if "IMU_Spike_Gen" in df.columns and "IMU_Spike_Det" in df.columns:
        analyze_and_plot_confusion(df["IMU_Spike_Gen"], df["IMU_Spike_Det"], "IMU")
        
    if "Cam_Outlier_Gen" in df.columns and "Cam_Outlier_Det" in df.columns:
        analyze_and_plot_confusion(df["Cam_Outlier_Gen"], df["Cam_Outlier_Det"], "Telecamera")

if __name__ == "__main__":
    # Puoi modificare le probabilità inserendo i valori desiderati come parametro
    # Esempio: 5% di probabilità per l'IMU e 3% per la Telecamera
    run_automated_analysis(imu_prob=0.4, cam_prob=0.4)