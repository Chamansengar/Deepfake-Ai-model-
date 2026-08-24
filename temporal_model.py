"""
Temporal Deepfake Detection Model — Hybrid spatial-temporal architecture.

Combines a frozen InceptionResnetV1 backbone (spatial face embeddings)
with a Transformer or LSTM temporal head to detect deepfakes from
video frame sequences.

Architecture:
    Video clip (B, T, 3, 256, 256)
        → InceptionResnetV1 per frame → (B, T, 512) embeddings
        → Temporal Head (Transformer / LSTM)
        → Binary classification (real/fake)
"""

import math
import os
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


# ============================================================
# Positional Encoding for Transformer
# ============================================================

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings for frame positions."""

    def __init__(self, max_len: int = 64, d_model: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            (B, T, D) tensor with positional encoding added
        """
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_embedding(positions)


# ============================================================
# Transformer Temporal Head
# ============================================================

class TransformerTemporalHead(nn.Module):
    """
    Transformer encoder for temporal analysis of frame embeddings.

    Uses a [CLS] token for sequence-level classification.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (max_seq_len + 1 for [CLS])
        self.pos_encoding = LearnablePositionalEncoding(max_seq_len + 1, d_model)

        # Layer norm before transformer
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 512) frame embeddings
        Returns:
            (B, 1) logits
        """
        B, T, D = x.shape

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.input_norm(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, T+1, D)

        # Use [CLS] token output for classification
        cls_output = x[:, 0, :]  # (B, D)

        return self.classifier(cls_output)  # (B, 1)


# ============================================================
# LSTM Temporal Head (Alternative)
# ============================================================

class LSTMTemporalHead(nn.Module):
    """
    Bidirectional LSTM for temporal analysis of frame embeddings.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head (bidirectional → 2 * hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 512) frame embeddings
        Returns:
            (B, 1) logits
        """
        output, (h_n, _) = self.lstm(x)  # output: (B, T, 2*hidden)

        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, B, hidden_dim)
        forward_last = h_n[-2, :, :]   # (B, hidden_dim)
        backward_last = h_n[-1, :, :]  # (B, hidden_dim)
        combined = torch.cat([forward_last, backward_last], dim=1)  # (B, 2*hidden)

        return self.classifier(combined)  # (B, 1)


# ============================================================
# Full Temporal Deepfake Model
# ============================================================

class TemporalDeepfakeModel(nn.Module):
    """
    Hybrid spatial-temporal model for video deepfake detection.

    Spatial backbone:  InceptionResnetV1 (pretrained VGGFace2, frozen)
    Temporal head:     Transformer encoder or bidirectional LSTM

    Args:
        temporal_head:    'transformer' or 'lstm'
        backbone_weights: Path to fine-tuned backbone checkpoint (optional).
                          If None, uses raw VGGFace2 pretrained weights.
        freeze_backbone:  If True, freeze all backbone parameters.
        device:           Target device.
        seq_len:          Maximum sequence length.
    """

    def __init__(
        self,
        temporal_head: str = 'transformer',
        backbone_weights: str = None,
        freeze_backbone: bool = True,
        device: str = 'cpu',
        seq_len: int = 16,
    ):
        super().__init__()
        self.device = device

        # ---- Spatial Backbone ----
        self.backbone = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,  # We want embeddings, not classification
            device=device,
        )

        # Load fine-tuned weights if provided
        if backbone_weights and os.path.isfile(backbone_weights):
            print(f"[TemporalModel] Loading backbone weights from: {backbone_weights}")
            checkpoint = torch.load(backbone_weights, map_location=device, weights_only=True)
            # The checkpoint was saved with classify=True, num_classes=1
            # We need to handle the extra classification layer
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Filter out the classification head keys
            backbone_state = {
                k: v for k, v in state_dict.items()
                if not k.startswith('logits') and not k.startswith('last_linear') and not k.startswith('last_bn')
            }
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"[TemporalModel] Backbone weights loaded successfully")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print(f"[TemporalModel] Backbone frozen (no gradients)")

        self.freeze_backbone = freeze_backbone

        # ---- Temporal Head ----
        embedding_dim = 512  # InceptionResnetV1 embedding size

        if temporal_head == 'transformer':
            self.temporal = TransformerTemporalHead(
                d_model=embedding_dim,
                nhead=4,
                num_layers=2,
                dim_feedforward=1024,
                dropout=0.3,
                max_seq_len=seq_len,
            )
            print(f"[TemporalModel] Using Transformer temporal head "
                  f"(2 layers, 4 heads, seq_len={seq_len})")
        elif temporal_head == 'lstm':
            self.temporal = LSTMTemporalHead(
                input_dim=embedding_dim,
                hidden_dim=256,
                num_layers=2,
                dropout=0.3,
            )
            print(f"[TemporalModel] Using LSTM temporal head (2 layers, bidirectional)")
        else:
            raise ValueError(f"Unknown temporal_head: '{temporal_head}'. Use 'transformer' or 'lstm'.")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TemporalModel] Total params: {total_params:,}")
        print(f"[TemporalModel] Trainable params: {trainable_params:,} "
              f"({trainable_params / total_params * 100:.1f}%)")

    def extract_embeddings(self, frames):
        """
        Extract per-frame embeddings from the backbone.

        Args:
            frames: (B, T, 3, H, W) tensor of face frames

        Returns:
            (B, T, 512) tensor of frame embeddings
        """
        B, T, C, H, W = frames.shape

        # Reshape to process all frames at once: (B*T, 3, H, W)
        frames_flat = frames.reshape(B * T, C, H, W)

        # Extract embeddings
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.backbone(frames_flat)  # (B*T, 512)
        else:
            embeddings = self.backbone(frames_flat)

        # Reshape back: (B, T, 512)
        embeddings = embeddings.reshape(B, T, -1)

        return embeddings

    def forward(self, frames):
        """
        Args:
            frames: (B, T, 3, H, W) tensor of face frame sequences

        Returns:
            (B, 1) logits (pass through sigmoid for probability)
        """
        embeddings = self.extract_embeddings(frames)  # (B, T, 512)
        logits = self.temporal(embeddings)  # (B, 1)
        return logits

    def train(self, mode=True):
        """Override to keep backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


# ============================================================
# Utility: Load model for inference
# ============================================================

def load_temporal_model(
    checkpoint_path: str,
    temporal_head: str = 'transformer',
    device: str = 'cpu',
    seq_len: int = 16,
):
    """
    Load a trained temporal model from checkpoint.

    Args:
        checkpoint_path: Path to the temporal model checkpoint.
        temporal_head: 'transformer' or 'lstm'.
        device: Target device.
        seq_len: Sequence length the model was trained with.

    Returns:
        Loaded model in eval mode.
    """
    model = TemporalDeepfakeModel(
        temporal_head=temporal_head,
        freeze_backbone=True,
        device=device,
        seq_len=seq_len,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # noqa: S614
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[TemporalModel] Loaded checkpoint from '{checkpoint_path}'")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    val_metrics = checkpoint.get('val_metrics', {})
    print(f"  Val AUC: {val_metrics.get('auc', '?')}")
    print(f"  Val Acc: {val_metrics.get('accuracy', '?')}")

    return model
