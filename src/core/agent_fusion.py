import torch
import torch.nn as nn

class AttentionFusionLayer(nn.Module):
    """Dynamic sensor weighing mechanism using scaled dot-product attention."""
    def __init__(self, input_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, sensor_embeddings):
        # sensor_embeddings shape: [Batch, NumSensors, Dim]
        q = self.query(sensor_embeddings)
        k = self.key(sensor_embeddings)
        v = self.value(sensor_embeddings)
        
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1)**0.5), dim=-1)
        fused_representation = torch.matmul(attn_weights, v)
        return fused_representation, attn_weights

class IntelligentSustainabilityAgent:
    """Orchestrates sensor fusion and anomaly detection for groundwater policy making."""
    def __init__(self):
        self.fusion_engine = AttentionFusionLayer()
        print("Intelligent Sustainability Agent Initialized.")
