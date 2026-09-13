# EKF con Rilevamento degli Outlier per Robot Uniciclo

## Panoramica
Questa repository contiene una simulazione basata su Python di un Filtro di Kalman Esteso (EKF) applicato a un robot mobile di tipo uniciclo. L'obiettivo principale di questo progetto è eseguire la stima dello stato (Localizzazione/SLAM) fondendo dati di sensori eterogenei (IMU propriocettiva e Telecamera esterocettiva) e gestendo in modo robusto gli outlier sensoriali e i picchi di rumore.

## Funzionalità Principali
*   **Cinematica dell'Uniciclo:** Modellazione matematica realistica di un robot uniciclo a guida differenziale, che offre un perfetto equilibrio tra realismo fisico e complessità matematica.
*   **Filtro di Kalman Esteso (EKF):** Fonde i dati IMU ad alta frequenza (fase di predizione) con i dati della telecamera a bassa frequenza (fase di aggiornamento) per stimare lo stato del robot e compensare la deriva nel tempo.
*   **Rigetto Robusto degli Outlier:**
    *   **IMU (Propriocettiva):** Utilizza un approccio statistico a finestra mobile (sliding-window) per calcolare medie e covarianze locali. Identifica correttamente i picchi anomali isolati del sensore, adattandosi al contempo alle reali manovre improvvise.
    *   **Telecamera (Esterocettiva):** Impiega un approccio basato sull'innovazione integrato direttamente con l'architettura predittiva dell'EKF. 
    *   Entrambi gli approcci sfruttano la **Distanza di Mahalanobis** e la sogliatura basata sul test del **Chi-Quadro** per scartare le misurazioni corrotte senza compromettere la stabilità del filtro.

## Struttura della Repository
*   `src/unicycle_estimation.py`: Script di simulazione principale contenente la cinematica dell'uniciclo, il ciclo di simulazione e la funzionalità di logging in CSV.
*   `src/kalman_filter.py`: Implementazione principale delle equazioni del Filtro di Kalman Esteso (fasi di Predizione e Aggiornamento).
*   `src/outliers_detection.py`: Contiene le classi `IMUOutlierRejector` e `CameraOutlierRejector` per il filtraggio robusto.
*   `src/plot.py`: Script per l'analisi dei log CSV generati e la creazione di metriche di performance visive (es. grafici di rilevamento dei picchi IMU, matrici di confusione).
*   `src/dati_sensori.csv`: Output di log generato dalla simulazione contenente i dati reali (ground truth), le stime e i flag di rilevamento degli outlier.
*   `report/`: Contiene il report accademico in formato Quarto (`.qmd`) che descrive in dettaglio le basi teoriche, la matematica dei sensori e i risultati finali.

## Per Iniziare
1.  Assicurati di avere Python 3.x installato insieme alle dipendenze necessarie (`numpy`, `scipy`, `matplotlib`).
2.  Esegui la simulazione principale per generare i log dei dati dei sensori:
    ```bash
    python src/unicycle_estimation.py
    ```
3.  Genera i risultati e i grafici per validare il rigetto degli outlier:
    ```bash
    python src/plot.py
    ```

## Teoria e Documentazione
Per un approfondimento sulla matematica alla base dell'EKF, sul modello dell'uniciclo e sulla logica della distanza di Mahalanobis utilizzata per l'Associazione dei Dati (Data Association) e il Rigetto degli Outlier, si prega di fare riferimento al report in Quarto situato nella cartella `report/`.
