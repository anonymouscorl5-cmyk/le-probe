import torch
import torch.nn as nn
import torch.nn.functional as F


class Transcoder(nn.Module):
    """
    Universal Transcoder / Crosscoder Architecture.

    Modes:
    - SAE: Identity mapping (Layer L -> Layer L). d_model == d_output.
    - Transcoder: Transition mapping (Layer L -> Layer L+1). d_model == d_output.
    - Crosscoder: Multi-layer mapping (Layer L -> Layers L, L+1, ... L+N).
      In this mode, d_output = d_model * num_target_layers.
    """

    def __init__(
        self,
        d_model: int,
        d_dict: int,
        d_output: int | None = None,
        l1_coeff: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.d_output = d_output if d_output is not None else d_model
        self.l1_coeff = l1_coeff

        # Encoder: Linear + Bias + ReLU
        # Probes the input space (Source Layer)
        self.encoder = nn.Linear(d_model, d_dict)
        self.b_enc = nn.Parameter(torch.zeros(d_dict))

        # Decoder: Linear
        # Projects back into the output space (Single or Multiple Layers)
        self.decoder = nn.Linear(d_dict, self.d_output, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(self.d_output))

        # Orthogonal initialization for stability
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x, target=None):
        """
        x: Input activations (Batch, d_model)
        target: Target activations for training (Optional, Batch, d_output)
        """
        # Centering
        x_centered = (
            x - self.b_dec[: self.d_model] if self.d_output == self.d_model else x
        )

        # Encode: Map to sparse latent space
        acts = F.relu(self.encoder(x_centered) + self.b_enc)

        # Decode: Map to target space
        x_hat = self.decoder(acts) + self.b_dec

        if target is not None:
            # Loss: MSE + Sparsity
            l2_loss = F.mse_loss(x_hat, target)
            l1_loss = acts.abs().sum(dim=-1).mean()
            total_loss = l2_loss + self.l1_coeff * l1_loss

            return {
                "output": x_hat,
                "activations": acts,
                "loss": total_loss,
                "l2_loss": l2_loss,
                "l1_loss": l1_loss,
            }

        return {"output": x_hat, "activations": acts}

    @torch.no_grad()
    def normalize_decoder(self):
        """Ensure dictionary atoms are unit norm."""
        W = self.decoder.weight.data
        W.div_(W.norm(dim=0, keepdim=True) + 1e-8)
