import numpy as np
from scipy.stats import chi2

class IMUOutlierRejector:
    def __init__(self, window_size=50, confidence_level=0.99, dof=3):
        self.window_size = window_size
        self.dof = dof
        self.chi2_threshold = chi2.ppf(confidence_level, df=self.dof)
        
        self.history = []
        
        # Contatore per distinguere uno spike isolato da una manovra reale
        self.consecutive_outliers = 0 

        self.outlier_detected = False

    def process(self, measurement):
        z = measurement.flatten()

        if len(self.history) < self.window_size:
            self.outlier_detected = False
            self.history.append(z)
            return measurement
        
        history_array = np.array(self.history)
        mu = np.mean(history_array, axis=0)
        
        Sigma = np.cov(history_array, rowvar=False)
        
        # senza far esplodere la Distanza di Mahalanobis.
        Sigma += np.eye(self.dof) * 5.0

        diff = z - mu
        mahalanobis_sq = diff.T @ np.linalg.inv(Sigma) @ diff

        if mahalanobis_sq > self.chi2_threshold:
            self.consecutive_outliers += 1
            self.outlier_detected = True
            print(f"[Outlier Rejector] Spike rilevato! D^2: {mahalanobis_sq:.2f} > Soglia: {self.chi2_threshold:.2f}")
            
            # NUOVO BLOCCO: Se vediamo 3 outlier di fila, è l'utente che guida, non un errore!
            if self.consecutive_outliers > 3:
                print("   -> [Attenzione] Troppi outlier consecutivi! È una manovra reale, resetto il filtro.")
                # Resettiamo la storia alla nuova dinamica
                self.history = [z]
                self.consecutive_outliers = 0
                return measurement

            # Ritorna la media storica per ignorare il singolo spike
            return mu.reshape((3,1))
            
        else:
            # DATO VALIDO: Azzeriamo il contatore e aggiorniamo la finestra
            self.outlier_detected = False
            self.consecutive_outliers = 0
            self.history.pop(0)
            self.history.append(z)
            return measurement
        


class CameraOutlierRejector:
    def __init__(self, confidence_level=0.99):
        self.dof = 1

        # Soglia del Chi-Quadro per 1 grado di libertà (pixel 1D)
        self.chi2_threshold = chi2.ppf(confidence_level, df=self.dof)
        self.outliers_rejected_count = 0

    def process(self, camera_measurements, ekf, delta_t):
        filtered_landmarks = []
        filtered_pixels = []
        
        # ANTICIPAZIONE DELL'EKF: 
        # Calcoliamo lo stato e la covarianza predetti esattamente come farà l'EKF 
        # all'inizio della sua funzione update()
        pred_state = ekf.state + ekf.d_state * delta_t
        pred_state[4, 0] = (pred_state[4, 0] + np.pi) % (2 * np.pi) - np.pi
        
        pred_P = ekf.P + ekf.P_dot * delta_t
        
        for lm, z_true in zip(camera_measurements['landmarks'], camera_measurements['pixels']):
            
            # MISURAZIONE ATTESA E JACOBIANO (H):
            # Usiamo direttamente il modello matematico interno all'EKF!
            z_pred, H_i = ekf.measurement_model(pred_state, lm)
            
            # COVARIANZA DELL'INNOVAZIONE (S):
            # Proiettiamo l'incertezza dello stato (pred_P) nello spazio dei pixel
            # e sommiamo il rumore del sensore (ekf.R)
            S_i = H_i @ pred_P @ H_i.T + ekf.R
            
            # Estraiamo il valore scalare in modo sicuro
            S_i_scalar = float(np.squeeze(S_i))
            
            # CALCOLO MAHALANOBIS
            diff = float(z_true - z_pred)
            mahalanobis_sq = (diff ** 2) / S_i_scalar
            
            # REJECTION
            if mahalanobis_sq <= self.chi2_threshold:
                filtered_landmarks.append(lm)
                filtered_pixels.append(z_true)
            else:
                self.outliers_rejected_count += 1
                print(f"[Camera Rejector] Scartato! Err: {diff:.2f}px | D^2: {mahalanobis_sq:.2f} > Soglia: {self.chi2_threshold:.2f} (Incertezza S: {S_i_scalar:.4f})")

        return {'landmarks': filtered_landmarks, 'pixels': filtered_pixels}