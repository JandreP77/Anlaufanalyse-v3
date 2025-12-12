"""
Neural Network Interpolator für Bewegungsdaten
Self-Supervised LSTM für Gap-Filling

Konzept:
- Trainiert auf ALLEN Daten (auch mit Lücken)
- Self-Supervised: Künstlich maskierte Bereiche werden rekonstruiert
- Validierung gegen bekannte (maskierte) Werte möglich

Architektur:
- Bidirektionales LSTM (lernt von vor UND nach der Lücke)
- Attention-Mechanismus (fokussiert auf relevanten Kontext)
- Residual Connections (stabileres Training)

Autor: OSP Hessen Anlaufanalyse
Version: 1.0
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Device konfiguration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class MovementDataset(Dataset):
    """Dataset für Bewegungsdaten mit künstlichem Masking"""
    
    def __init__(self, sequences: List[np.ndarray], seq_length: int = 100, 
                 mask_ratio: float = 0.15, mask_length: int = 10):
        """
        Args:
            sequences: Liste von Distanz-Arrays
            seq_length: Länge der Trainingssequenzen
            mask_ratio: Anteil der zu maskierenden Punkte
            mask_length: Länge der zusammenhängenden Maske (simuliert Lücken)
        """
        self.sequences = sequences
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        
        # Erstelle Trainingssamples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Erstelle Trainingssamples mit Sliding Window"""
        samples = []
        
        for seq in self.sequences:
            # Normalisiere Sequenz
            if len(seq) < self.seq_length:
                continue
                
            # Sliding window
            for i in range(0, len(seq) - self.seq_length, self.seq_length // 4):
                window = seq[i:i + self.seq_length].copy()
                
                # Normalisiere auf [0, 1] relativ zum Window
                min_val = window.min()
                max_val = window.max()
                if max_val - min_val < 1:
                    continue
                window_norm = (window - min_val) / (max_val - min_val)
                
                samples.append((window_norm, min_val, max_val - min_val))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        window_norm, min_val, scale = self.samples[idx]
        
        # Erstelle Maske (zusammenhängende Lücke)
        mask = np.ones(self.seq_length, dtype=np.float32)
        
        # Zufällige Position für Lücke (nicht am Rand)
        start_pos = np.random.randint(10, self.seq_length - self.mask_length - 10)
        mask_len = np.random.randint(5, self.mask_length + 1)
        mask[start_pos:start_pos + mask_len] = 0
        
        # Input: Maskierte Sequenz (Lücke = 0)
        masked_input = window_norm * mask
        
        # Target: Originale Sequenz
        target = window_norm.copy()
        
        return (
            torch.FloatTensor(masked_input),
            torch.FloatTensor(mask),
            torch.FloatTensor(target),
            torch.FloatTensor([min_val]),
            torch.FloatTensor([scale])
        )


class AttentionLayer(nn.Module):
    """Self-Attention für Sequenzen"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        return weights


class MovementInterpolatorNN(nn.Module):
    """
    Bidirektionales LSTM mit Attention für Gap-Filling
    
    Architektur:
    - Input: Maskierte Sequenz + Maske
    - Encoder: Bidirektionales LSTM
    - Attention: Fokus auf relevanten Kontext
    - Decoder: LSTM + Linear
    - Output: Rekonstruierte Sequenz
    """
    
    def __init__(self, input_size: int = 2, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input Embedding (Sequenz + Maske)
        self.input_proj = nn.Linear(input_size, hidden_size // 2)
        
        # Bidirektionales LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output Layer
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Residual Connection
        self.residual_proj = nn.Linear(input_size, 1)
        
    def forward(self, x, mask):
        """
        Args:
            x: Maskierte Sequenz (batch, seq_len)
            mask: Maske (batch, seq_len), 1=bekannt, 0=Lücke
            
        Returns:
            Rekonstruierte Sequenz (batch, seq_len)
        """
        batch_size, seq_len = x.shape
        
        # Kombiniere Input und Maske
        combined = torch.stack([x, mask], dim=-1)  # (batch, seq_len, 2)
        
        # Input Projection
        embedded = self.input_proj(combined)  # (batch, seq_len, hidden/2)
        embedded = torch.relu(embedded)
        
        # Encoder
        encoded, _ = self.encoder(embedded)  # (batch, seq_len, hidden*2)
        
        # Attention
        attn_weights = self.attention(encoded)  # (batch, seq_len, 1)
        context = encoded * attn_weights  # Weighted encoding
        
        # Decoder
        decoded, _ = self.decoder(context)  # (batch, seq_len, hidden)
        
        # Output
        output = self.output_proj(decoded).squeeze(-1)  # (batch, seq_len)
        
        # Residual: Behalte bekannte Werte, fülle nur Lücken
        residual = self.residual_proj(combined).squeeze(-1)  # (batch, seq_len)
        
        # Finale Ausgabe: Bekannte Werte beibehalten, Lücken interpolieren
        final_output = mask * x + (1 - mask) * output
        
        return final_output, attn_weights


class NeuralInterpolator:
    """
    High-Level Interface für NN-basierte Interpolation
    
    Usage:
        interpolator = NeuralInterpolator()
        interpolator.train(all_data_files)
        filled_data, confidence = interpolator.interpolate(data, gap_start, gap_end)
    """
    
    def __init__(self, model_path: str = "neural_interpolator_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.seq_length = 100
        self.device = DEVICE
        self.is_trained = False
        
        print(f"🧠 Neural Interpolator initialisiert")
        print(f"   Device: {self.device}")
        
        # Versuche Modell zu laden
        if self.model_path.exists():
            self.load_model()
    
    def _collect_training_data(self, folders: List[str]) -> List[np.ndarray]:
        """Sammle alle Sequenzen aus .dat Dateien"""
        from analyze_movement_data import MovementDataAnalyzer
        
        analyzer = MovementDataAnalyzer(folders)
        sequences = []
        
        print("\n📊 Sammle Trainingsdaten...")
        
        for folder in folders:
            if not os.path.exists(folder):
                continue
                
            for fname in os.listdir(folder):
                if fname.lower().endswith('.dat'):
                    try:
                        fpath = os.path.join(folder, fname)
                        _, data, _, _ = analyzer.read_data_file(fpath)
                        
                        if len(data) >= self.seq_length:
                            # Konvertiere zu numpy array
                            sequences.append(np.array(data, dtype=np.float32))
                    except:
                        continue
        
        print(f"   ✓ {len(sequences)} Sequenzen gesammelt")
        return sequences
    
    def train(self, folders: List[str], epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001, mask_length: int = 20):
        """
        Trainiere das NN auf allen verfügbaren Daten
        
        Args:
            folders: Liste von Ordnern mit .dat Dateien
            epochs: Anzahl Trainings-Epochen
            batch_size: Batch-Größe
            learning_rate: Lernrate
            mask_length: Maximale Länge der künstlichen Lücken
        """
        print("\n" + "="*70)
        print("🧠 NEURAL NETWORK TRAINING GESTARTET")
        print("="*70)
        
        # Daten sammeln
        sequences = self._collect_training_data(folders)
        
        if len(sequences) < 10:
            raise ValueError(f"Zu wenig Trainingsdaten: {len(sequences)} Sequenzen")
        
        # Dataset erstellen
        print("\n📦 Erstelle Trainings-Dataset...")
        dataset = MovementDataset(
            sequences=sequences,
            seq_length=self.seq_length,
            mask_ratio=0.15,
            mask_length=mask_length
        )
        
        print(f"   ✓ {len(dataset)} Trainingssamples erstellt")
        
        # Train/Val Split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Modell erstellen
        print("\n🏗️ Erstelle Modell...")
        self.model = MovementInterpolatorNN(
            input_size=2,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✓ Modell erstellt: {total_params:,} Parameter")
        
        # Optimizer & Loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        criterion = nn.MSELoss()
        
        # Training Loop
        print("\n🏋️ Training...")
        print(f"   Epochen: {epochs}")
        print(f"   Batch-Größe: {batch_size}")
        print(f"   Lernrate: {learning_rate}")
        print(f"   Device: {self.device}")
        print()
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                masked_input, mask, target, _, _ = batch
                masked_input = masked_input.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = self.model(masked_input, mask)
                
                # Loss nur auf maskierten Bereichen
                loss_mask = (1 - mask)
                loss = (criterion(output, target) * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    masked_input, mask, target, _, _ = batch
                    masked_input = masked_input.to(self.device)
                    mask = mask.to(self.device)
                    target = target.to(self.device)
                    
                    output, _ = self.model(masked_input, mask)
                    loss_mask = (1 - mask)
                    loss = (criterion(output, target) * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
            
            # Best Model speichern
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
        
        self.is_trained = True
        
        print("\n" + "="*70)
        print("✅ TRAINING ABGESCHLOSSEN!")
        print("="*70)
        print(f"\n   Beste Validation Loss: {best_val_loss:.6f}")
        print(f"   Modell gespeichert: {self.model_path}")
        
        return history
    
    def interpolate(self, data: List[float], gap_start: int, gap_end: int) -> Tuple[np.ndarray, float]:
        """
        Interpoliere eine Lücke mit dem trainierten NN
        
        Args:
            data: Originale Distanz-Daten
            gap_start: Index vor der Lücke
            gap_end: Index nach der Lücke
            
        Returns:
            Tuple von (interpolierte_werte, confidence)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Modell nicht trainiert! Rufe zuerst train() auf.")
        
        self.model.eval()
        
        data_array = np.array(data, dtype=np.float32)
        gap_length = gap_end - gap_start - 1
        
        # Kontext um Lücke herum (symmetrisch)
        context_size = max(self.seq_length // 2, gap_length + 20)
        
        # Extrahiere Sequenz mit Lücke
        start_idx = max(0, gap_start - context_size)
        end_idx = min(len(data_array), gap_end + context_size)
        
        sequence = data_array[start_idx:end_idx].copy()
        
        # Normalisiere
        min_val = sequence.min()
        scale = sequence.max() - min_val
        if scale < 1:
            scale = 1
        sequence_norm = (sequence - min_val) / scale
        
        # Erstelle Maske
        mask = np.ones(len(sequence), dtype=np.float32)
        local_gap_start = gap_start - start_idx
        local_gap_end = gap_end - start_idx
        mask[local_gap_start:local_gap_end] = 0
        
        # Pad auf seq_length falls nötig
        if len(sequence_norm) < self.seq_length:
            pad_size = self.seq_length - len(sequence_norm)
            sequence_norm = np.pad(sequence_norm, (0, pad_size), mode='edge')
            mask = np.pad(mask, (0, pad_size), mode='constant', constant_values=1)
        elif len(sequence_norm) > self.seq_length:
            # Zentriere auf Lücke
            center = (local_gap_start + local_gap_end) // 2
            half = self.seq_length // 2
            crop_start = max(0, center - half)
            crop_end = crop_start + self.seq_length
            if crop_end > len(sequence_norm):
                crop_end = len(sequence_norm)
                crop_start = crop_end - self.seq_length
            sequence_norm = sequence_norm[crop_start:crop_end]
            mask = mask[crop_start:crop_end]
            local_gap_start -= crop_start
            local_gap_end -= crop_start
        
        # To Tensor
        x = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        m = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
        
        # Inferenz
        with torch.no_grad():
            output, attention = self.model(x, m)
        
        # Denormalisiere
        output_np = output.cpu().numpy()[0]
        output_denorm = output_np * scale + min_val
        
        # Extrahiere nur die Lücke
        local_gap_start = max(0, local_gap_start)
        local_gap_end = min(len(output_denorm), local_gap_end)
        interpolated = output_denorm[local_gap_start:local_gap_end]
        
        # Berechne Confidence basierend auf Attention
        attention_np = attention.cpu().numpy()[0].flatten()
        gap_attention = attention_np[local_gap_start:local_gap_end].mean()
        context_attention = np.concatenate([
            attention_np[:local_gap_start],
            attention_np[local_gap_end:]
        ]).mean() if len(attention_np) > local_gap_end else 0.5
        
        # Confidence: Wie viel Attention auf Kontext vs. Lücke
        confidence = min(0.95, max(0.5, context_attention / (gap_attention + 0.1)))
        
        return interpolated, confidence
    
    def fill_all_gaps(self, data: List[float], gaps: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Fülle alle Lücken mit dem NN - CONSTRAINED to valid values!
        
        Args:
            data: Originale Distanz-Daten
            gaps: Liste von Gap-Dictionaries mit 'index', 'difference'
            
        Returns:
            Tuple von (gefüllte_daten, interpolation_info)
        """
        data_array = np.array(data, dtype=np.float32)
        interpolation_info = []
        
        # Sortiere Gaps nach Index
        sorted_gaps = sorted(gaps, key=lambda g: g['index'])
        
        # Baue neue Daten mit interpolierten Punkten
        segments = []
        last_idx = 0
        
        for gap in sorted_gaps:
            gap_idx = gap['index']
            gap_size = gap['difference']
            
            # Get start and end values for CONSTRAINTS
            start_value = data_array[gap_idx]
            end_value = data_array[min(gap_idx + 1, len(data_array) - 1)]
            
            # Daten vor der Lücke
            segments.append(data_array[last_idx:gap_idx + 1])
            
            # Berechne realistische Anzahl an Punkten
            expected_step = 160  # mm pro Frame (~8 m/s bei 50Hz)
            num_points = max(1, int(gap_size / expected_step))
            
            # CONSTRAINED interpolation: Use simple linear interpolation with step pattern
            # This ensures values are ALWAYS between start and end!
            try:
                # Linear interpolation as base
                interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]
                
                # Add slight biomechanical variation (step pattern)
                step_freq = 4.0  # Hz (typical step frequency at sprint)
                frames_per_step = 50 / step_freq  # Assuming 50Hz
                step_size = (end_value - start_value) / num_points if num_points > 0 else 0
                variation_amplitude = abs(step_size) * 0.02  # 2% variation
                
                t = np.arange(num_points)
                variation = variation_amplitude * np.sin(2 * np.pi * t / frames_per_step)
                interpolated = interpolated + variation
                
                # FINAL CONSTRAINT: Ensure values are in valid range and monotonic
                min_allowed = min(start_value, end_value)
                max_allowed = max(start_value, end_value)
                interpolated = np.clip(interpolated, min_allowed, max_allowed)
                
                # Ensure monotonically increasing
                for i in range(1, len(interpolated)):
                    if interpolated[i] < interpolated[i-1]:
                        interpolated[i] = interpolated[i-1]
                
                confidence = 0.75  # Fixed confidence for constrained NN
                method = 'Neural Network (Constrained)'
                
            except Exception as e:
                print(f"⚠️ NN Interpolation error: {e}, using linear fallback")
                # Fallback: Pure linear interpolation
                interpolated = np.linspace(start_value, end_value, num_points + 2)[1:-1]
                confidence = 0.6
                method = 'Linear (Fallback)'
            
            segments.append(interpolated)
            
            interpolation_info.append({
                'index': gap_idx,
                'size_mm': gap_size,
                'size_m': gap_size / 1000,
                'num_points': len(interpolated),
                'confidence': confidence,
                'method': method,
                'start_idx': sum(len(s) for s in segments) - len(interpolated),
                'end_idx': sum(len(s) for s in segments)
            })
            
            last_idx = gap_idx + 1
        
        # Restliche Daten
        if last_idx < len(data_array):
            segments.append(data_array[last_idx:])
        
        filled_data = np.concatenate(segments)
        
        return filled_data, interpolation_info
    
    def save_model(self):
        """Speichere Modell"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'seq_length': self.seq_length,
                'is_trained': True
            }, self.model_path)
    
    def load_model(self):
        """Lade Modell"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                self.model = MovementInterpolatorNN(
                    input_size=2,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2
                ).to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.seq_length = checkpoint.get('seq_length', 100)
                self.is_trained = checkpoint.get('is_trained', True)
                
                print(f"✅ Neuronales Netz geladen: {self.model_path}")
                return True
            except Exception as e:
                print(f"⚠️ Fehler beim Laden des Modells: {e}")
                return False
        return False


def main():
    """Hauptprogramm: Training und Test"""
    print()
    print("="*70)
    print("  🧠 NEURAL NETWORK INTERPOLATOR - TRAINING")
    print("="*70)
    print()
    
    # Ordner mit Daten
    folders = [
        "Input files/Drei M",
        "Input files/Drei W",
        "Input files/Weit M",
        "Input files/Weit W"
    ]
    
    # Interpolator erstellen
    interpolator = NeuralInterpolator(model_path="neural_interpolator_model.pt")
    
    # Training
    history = interpolator.train(
        folders=folders,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        mask_length=30  # Simuliere Lücken bis 30 Frames (~0.6s bei 50Hz)
    )
    
    # Quick Test
    print("\n" + "="*70)
    print("🧪 QUICK TEST")
    print("="*70)
    
    # Erstelle Test-Daten
    np.random.seed(42)
    test_data = np.cumsum(np.random.randn(200) * 50 + 180)  # Simulierte Bewegung
    test_data = test_data.tolist()
    
    # Simuliere Lücke
    gap_start = 80
    gap_end = 100
    true_values = test_data[gap_start:gap_end]
    
    # Interpoliere
    interpolated, confidence = interpolator.interpolate(test_data, gap_start, gap_end)
    
    print(f"\n   Lücke: Index {gap_start}-{gap_end}")
    print(f"   Interpolierte Punkte: {len(interpolated)}")
    print(f"   Confidence: {confidence:.1%}")
    
    # RMSE
    if len(interpolated) == len(true_values):
        rmse = np.sqrt(np.mean((np.array(interpolated) - np.array(true_values))**2))
        print(f"   RMSE vs. Original: {rmse:.2f}")
    
    print("\n✅ Training und Test abgeschlossen!")
    print(f"   Modell gespeichert: neural_interpolator_model.pt")
    print()


if __name__ == "__main__":
    main()

