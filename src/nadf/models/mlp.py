"""
MLP models for triplet learning, regression, and binary classification.
"""

import torch
import torch.nn as nn

from nadf.utils import get_activation


class TripletMLP(nn.Module):
    """
    A simple MLP to be trained with triplet loss. It learns a new embedding.
    """

    def __init__(self, input_dim, width=64, depth=3, output_dim=32, activation="relu", bias=True):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        current_dim = input_dim
        for _i in range(depth - 1):
            layers.append(nn.Linear(current_dim, width, bias=bias))
            layers.append(get_activation(activation))
            current_dim = width

        layers.append(nn.Linear(width, output_dim, bias=bias))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class RegressionMLP(TripletMLP):
    """
    A simple MLP to be trained with regression loss.
    """

    def __init__(self, input_dim, width=64, depth=3, activation="relu", bias=True):
        super().__init__(output_dim=1, input_dim=input_dim, width=width, depth=depth, activation=activation, bias=bias)

    #! Fix: Flatten the output
    def forward(self, x):
        output = super().forward(x)
        return output.flatten()


class BinaryClassificationMLP(RegressionMLP):
    """
    MLP for binary classification with BCE loss.
    """

    def forward(self, x):
        output = super().forward(x)
        return torch.sigmoid(output).flatten()  # Probabilities: 0 to 1


# ? IDEAS FOR ATTENTION RPOBES
class TransformerAttentionProbe(nn.Module):
    """
    Transformer-style multi-head attention probe for vision features.
    Treats each feature dimension as a token and applies full self-attention.
    """

    def __init__(self, input_dim=128, num_heads=4, hidden_dim=64, depth=2, activation="relu", bias=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim  # 128
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim  # ? HIDDEN DIM: COMPRESSION OR EXPANSION? E.G.: COULD DO 256 OR 512

        # Initial projection to embed each dimension
        self.input_projection = nn.Linear(1, hidden_dim, bias=bias)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, bias=bias, batch_first=True
                )
                for _ in range(depth)
            ]
        )

        # Layer normalization for each attention layer
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])

        # forward passes after each attention layer
        self.feed_forwards = nn.ModuleList(
            [
                # Feed-forward networks after each attention layer (standard 4x expansion), expand, nonliearity, compress back
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4, bias=bias),  # = nn.Linear(64, 256)
                    get_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim, bias=bias),  # = nn.Linear(256, 64)
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

        # Final layer norms for feed-forward
        self.ff_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])

        # Output projection: aggregate sequence to single value
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=bias),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=bias),
        )

    def forward(self, x):
        """
        Args:
            x: Input features of shape [batch_size, input_dim]
        Returns:
            output: Scalar output for each sample [batch_size]
        """
        batch_size = x.shape[0]

        # Reshape: treat each dimension as a token
        # [batch_size, input_dim] -> [batch_size, input_dim, 1]
        x = x.unsqueeze(-1)

        # Project each dimension to hidden_dim
        # [batch_size, input_dim, 1] -> [batch_size, input_dim, hidden_dim]
        x = self.input_projection(x)

        # Apply transformer layers
        for attn, ln1, ff, ln2 in zip(self.attention_layers, self.layer_norms, self.feed_forwards, self.ff_layer_norms):
            # Self-attention with residual connection
            attn_out, _ = attn(x, x, x)  # Query, Key, Value all from x
            x = ln1(x + attn_out)  # Residual + LayerNorm

            # Feed-forward with residual connection
            ff_out = ff(x)
            x = ln2(x + ff_out)  # Residual + LayerNorm

        # Aggregate across the sequence dimension
        # Option 1: Mean pooling
        # x = x.mean(dim=1)  # [batch_size, hidden_dim]

        # Option 2: Max pooling
        # x = x.max(dim=1)[0]  # [batch_size, hidden_dim]

        # Option 3: Attention pooling (learnable aggregation)
        pooling_weights = self.output_projection(x).squeeze(-1)  # [batch_size, input_dim]
        pooling_weights = torch.softmax(pooling_weights, dim=1)  # [batch_size, input_dim]
        x = torch.sum(x * pooling_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]

        # Final projection to scalar
        output = self.output_projection(x).squeeze(-1)  # [batch_size]

        return output
