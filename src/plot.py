import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_outlier_performance(csv_path="dati_sensori.csv"):
    # Carica i log della simulazione salvati da unicycle_estimation.py
    df = pd.read_csv(csv_path)
    
    # ---------------------------------------------------------
    # FIGURA 1: Analisi Segnale IMU e Detection degli Spike
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot del segnale grezzo
    ax1.plot(df['Frame'], df['IMU_Accel_X'], label='IMU Accel X (Raw)', color='gray', alpha=0.7)
    
    # Trova i frame dove sono stati generati e rilevati gli spike
    spike_gen = df[df['IMU_Spike_Gen'] == True]
    spike_det = df[df['IMU_Spike_Det'] == True]
    
    # Sovrapponi i marker
    ax1.scatter(spike_gen['Frame'], spike_gen['IMU_Accel_X'], 
                color='red', marker='o', s=50, label='Spike Generato (Ground Truth)')
    ax1.scatter(spike_det['Frame'], spike_det['IMU_Accel_X'], 
                color='green', marker='x', s=80, label='Spike Rilevato (Rejector)')
    
    ax1.set_title("Efficacia del Filtro Mahalanobis sull'IMU")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Accelerazione X [m/s^2]")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()

    # ---------------------------------------------------------
    # FIGURA 2: Errore di Stima ed Effetto degli Outlier Camera
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # Calcolo dell'errore Euclideo sulla posizione 2D
    df['Pos_Error'] = np.sqrt((df['True_X'] - df['Est_X'])**2 + (df['True_Y'] - df['Est_Y'])**2)
    
    ax2.plot(df['Frame'], df['Pos_Error'], label='Errore di Posizione (EKF)', color='blue')
    
    # Evidenziamo i momenti in cui la telecamera ha generato outlier
    cam_outliers = df[df['Cam_Outlier_Gen'] == True]['Frame']
    for frame in cam_outliers:
        ax2.axvline(x=frame, color='red', alpha=0.3, linestyle='--')
        
    # Hack per aggiungere la linea rossa alla legenda una sola volta
    ax2.axvline(x=-1, color='red', alpha=0.3, linestyle='--', label='Outlier Camera Iniettato')
    
    ax2.set_xlim(df['Frame'].min(), df['Frame'].max())
    ax2.set_title("Errore di Stima della Posizione EKF")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Errore [m]")
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()

    plt.show()

if __name__ == "__main__":
    plot_outlier_performance()