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

        diff = z - mu
        mahalanobis_sq = diff.T @ np.linalg.inv(Sigma) @ diff

        if mahalanobis_sq > self.chi2_threshold:
            self.consecutive_outliers += 1
            self.outlier_detected = True
            print(f"[Outlier Rejector] Spike rilevato! D^2: {mahalanobis_sq:.2f} > Soglia: {self.chi2_threshold:.2f}")
            
            # Controllo che non ci siano stati troppi outliers consecutivi
            if self.consecutive_outliers > 3:
                print("   -> [Attenzione] Molti outlier consecutivi.")
                # Resettiamo la storia alla nuova dinamica
                self.history = [z]
                self.consecutive_outliers = 0
                return measurement

            # Ritorna la media storica per ignorare il singolo spike
            return mu.reshape((3,1))
            
        else:
            # Dato valido
            self.outlier_detected = False
            self.consecutive_outliers = 0
            self.history.pop(0)
            self.history.append(z)
            return measurement
        


class CameraOutlierRejector:
    def __init__(self, confidence_level=0.99):
        self.dof = 1

        # Calcolo della soglia del Chi-Quadro 
        self.chi2_threshold = chi2.ppf(confidence_level, df=self.dof)
        self.outliers_rejected_count = 0

    def process(self, camera_measurements, ekf, delta_t):
        filtered_landmarks = []
        filtered_pixels = []
        
        # Predizione dello stato e la covarianza come e' fatto nell'EKF
        pred_state = ekf.state + ekf.d_state * delta_t
        pred_state[4, 0] = (pred_state[4, 0] + np.pi) % (2 * np.pi) - np.pi
        
        pred_P = ekf.P + ekf.P_dot * delta_t
        
        for lm, z_true in zip(camera_measurements['landmarks'], camera_measurements['pixels']):

            # Misurazione attesa e Jacobiano
            z_pred, H_i = ekf.measurement_model(pred_state, lm)

            # Calcolo della covarianza dell'innovazione: Proiettando l'incertezza dello stato (pred_P) nello spazio dei pixel e sommiamo il rumore del sensore (ekf.R)
            S_i = H_i @ pred_P @ H_i.T + ekf.R
            
            # Estraiamo il valore scalare 
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
                print(f"[Camera Rejector] Dato Scartato! Err: {diff:.2f}px | D^2: {mahalanobis_sq:.2f} > Soglia: {self.chi2_threshold:.2f} (Incertezza S: {S_i_scalar:.4f})")

        return {'landmarks': filtered_landmarks, 'pixels': filtered_pixels}
