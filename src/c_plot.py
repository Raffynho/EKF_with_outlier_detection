import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# 1. Caricamento dei dati dal file CSV generato dalla simulazione
csv_file = "dati_sensori.csv"  # Modifica il percorso se necessario

try:
  df = pd.read_csv(csv_file)
  print(f"File '{csv_file}' caricato con successo. Righe totali: {len(df)}")
except FileNotFoundError:
  print(
      f"Errore: Il file '{csv_file}' non è stato trovato. Assicurati di aver"
      " eseguito prima la simulazione."
  )
  exit()


def analyze_and_plot_confusion(y_true, y_pred, sensor_name):
  """Calcola la matrice di confusione, i parametri statistici

  e genera un grafico visivo della matrice per il sensore specificato.
  """
  # Conversione in interi (0 o 1) per sicurezza
  y_true = y_true.astype(int)
  y_pred = y_pred.astype(int)

  # Calcolo della matrice di confusione [ [TN, FP], [FN, TP] ]
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  tn, fp, fn, tp = cm.ravel()

  # Calcolo dei parametri rilevanti
  accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = (
      (2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0
      else 0
  )

  print(f"\n--- Risultati Analisi: {sensor_name} ---")
  print(f"Veri Negativi (TN): {tn}")
  print(f"Falsi Positivi (FP): {fp} (Filtro troppo aggressivo)")
  print(f"Falsi Negativi (FN): {fn} (Anomalie non rilevate)")
  print(f"Veri Positivi (TP): {tp}")
  print(f"----------------------------------------")
  print(f"Accuratezza (Accuracy):  {accuracy:.4f}")
  print(f"Precisione (Precision):  {precision:.4f}")
  print(f"Richiamo (Recall):       {recall:.4f}")
  print(f"F1-Score:                {f1:.4f}")

  # Generazione del grafico della Matrice di Confusione
  plt.figure(figsize=(6, 5))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      cbar=False,
      xticklabels=["Normale (0)", "Outlier (1)"],
      yticklabels=["Normale (0)", "Outlier (1)"],
  )
  plt.title(f"Matrice di Confusione - Filtro Outlier ({sensor_name})")
  plt.xlabel("Valore Predetto dal Filtro")
  plt.ylabel("Ground Truth (Reale)")

  # Aggiunta di annotazioni testuali con i parametri nel grafico
  plt.figtext(
      0.15,
      0.02,
      f"Acc: {accuracy:.3f} | Prec: {precision:.3f} | Recall:"
      f" {recall:.3f} | F1: {f1:.3f}",
      fontsize=10,
      bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
  )

  plt.tight_layout()
  plt.savefig(
      f"matrice_confusione_{sensor_name.lower()}.png", dpi=300
  )  # Salva l'immagine per il report
  print(
      f"Grafico salvato come: matrice_confusione_{sensor_name.lower()}.png"
  )
  plt.show()


# 2. Analisi per il sensore IMU (se le colonne sono presenti nel CSV)
if "IMU_Spike_Gen" in df.columns and "IMU_Spike_Det" in df.columns:
  analyze_and_plot_confusion(df["IMU_Spike_Gen"], df["IMU_Spike_Det"], "IMU")
else:
  print("Colonne IMU_Spike_Gen / IMU_Spike_Det non trovate nel CSV.")

# 3. Analisi per il sensore Telecamera (se le colonne sono presenti nel CSV)
if "Cam_Outlier_Gen" in df.columns and "Cam_Outlier_Det" in df.columns:
  analyze_and_plot_confusion(
      df["Cam_Outlier_Gen"], df["Cam_Outlier_Det"], "Telecamera"
  )
else:
  print("Colonne Cam_Outlier_Gen / Cam_Outlier_Det non trovate nel CSV.")