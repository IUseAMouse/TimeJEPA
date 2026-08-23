"""
Transformer Predictor for JEPA.

This is a lightweight transformer that predicts target representations
from context representations. It's intentionally smaller than the encoder
(JEPA principle: predictor should be simpler than encoder).

Architecture: 4 layers vs 24 layers in encoder
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..components.attention import TransformerBlock


class TransformerPredictor(nn.Module):
    """
    Lightweight transformer predictor for JEPA.
    
    Takes context representations and predicts target representations.
    Key JEPA design: predictor is lighter than encoder (4 vs 24 layers).
    
    Architecture:
        Context embeddings [B, N_context, d_model]
        → Positional tokens for targets [B, N_target, d_model]
        → Concat [B, N_context + N_target, d_model]
        → Transformer blocks (4 layers)
        → Extract target predictions [B, N_target, d_model]
    
    Args:
        d_model: Model dimension (512)
        num_layers: Number of transformer layers (4, lighter than encoder)
        num_heads: Number of attention heads (8)
        d_ff: Feed-forward dimension (2048)
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_target_patches: int = 16,
        # G9.2 — conditionnement d'échelle w = k2/k1 (JEPA inter-résolution).
        # OPT-IN À LA CONSTRUCTION : sans ce flag l'attribut w_film n'existe
        # pas, donc le state_dict de toutes les configs existantes est inchangé
        # au bit près (leurs checkpoints se rechargent à l'identique).
        use_w_film: bool = False,
        # ESJEPA — voie z (statistiques du résidu, hétéroscédasticité
        # conditionnelle). Même contrat opt-in que use_w_film : flag off ⇒
        # l'attribut z_head n'existe pas, state_dict bit-identique.
        error_signal: bool = False,
        z_dim: int = 4,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        if use_w_film:
            # FiLM résiduel sur les requêtes futures : q · (1 + γ(log₂w)) + β(log₂w).
            # Poids ET biais initialisés à ZÉRO → γ=β=0 → identité exacte pour
            # tout w à l'initialisation. Deux conséquences voulues : (a) le
            # début d'entraînement de l'arm se comporte comme la baseline, le
            # conditionnement n'apparaît que si le gradient le demande ; (b) un
            # checkpoint xres rechargé SANS passer w (finetune, forecast) est
            # exactement le modèle à w=1 — aucun régime jamais vu.
            self.w_film = nn.Linear(1, 2 * d_model)
            nn.init.zeros_(self.w_film.weight)
            nn.init.zeros_(self.w_film.bias)

        if error_signal:
            # ESJEPA — tête z sur le TRONC du prédicteur : lit les tokens
            # cibles post-final_norm (AVANT prediction_head, qui est la tête de
            # la voie signal) et prédit les statistiques du résidu par patch
            # [B, N_target, z_dim]. Les gradients de la loss z remontent dans
            # le tronc et l'encodeur : c'est le mécanisme voulu — la
            # représentation apprend à retenir l'information de dispersion —
            # dosé par lambda_z côté module. Pas de BatchNorm (l'update EMA du
            # target encoder saute num_batches_tracked ; sans rapport ici mais
            # la contrainte est de famille : LayerNorm uniquement).
            self.z_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, z_dim),
            )

        self.future_position_embedding = nn.Parameter(
            torch.randn(1, max_target_patches, d_model) * 0.02
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                causal=False  # Non-causal (can attend to all positions)
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Optional: prediction head (linear projection)
        # This can help with representation quality
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict target representations from context.
        
        Args:
            context_embeddings: Context representations [B, N_context, d_model]
            target_positions: Target position indices [B, N_target]
                             These indicate which positions to predict
            attention_mask: Optional attention mask
            
        Returns:
            Predicted target representations [B, N_target, d_model]
        """
        batch_size = context_embeddings.shape[0]
        num_targets = target_positions.shape[1]

        future_queries = self._future_queries(batch_size, num_targets)

        x = torch.cat([context_embeddings, future_queries], dim=1)
        # x: [B, N_context + N_target, d_model]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Extract only the target predictions (last N_target tokens)
        target_predictions = x[:, -num_targets:, :]
        
        # Apply prediction head
        target_predictions = self.prediction_head(target_predictions)
        
        return target_predictions
    
    def _future_queries(self, batch_size: int, num_targets: int) -> torch.Tensor:
        """
        Fetch `num_targets` learned future queries, refusing to truncate.

        Slicing `future_position_embedding[:, :num_targets]` past the table size
        silently returns fewer rows. The output shape stayed correct downstream
        because the target slice `x[:, -num_targets:]` reads from the
        concatenated sequence — so the missing queries were quietly replaced by
        the LAST CONTEXT EMBEDDINGS, and those were then trained and scored as
        if they were predictions.

        With patch=16 / stride=8 that silently corrupted every configuration
        with prediction_length > 136 (e.g. large.yaml at 192 -> 23 target
        patches, 7 of them fake), and base.yaml at patch=4 / stride=4 -> 32
        target patches, half of them fake.
        """
        available = self.future_position_embedding.shape[1]
        if num_targets > available:
            raise ValueError(
                f"TransformerPredictor was built with max_target_patches="
                f"{available} but {num_targets} target patches were requested. "
                f"Increase max_target_patches (JEPATST sizes it from "
                f"prediction_length). Truncating here would silently feed "
                f"context embeddings in place of predictions."
            )
        return self.future_position_embedding[:, :num_targets, :].expand(batch_size, -1, -1)

    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        attention_mask: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        return_z: bool = False,
    ) -> torch.Tensor:
        """
        Simplified forward pass when target positions are just 'next N'.

        `w` (optionnel, [B]) : ratio d'échelle k2/k1 par ITEM (G9.2) — par item
        et non par batch, parce que la résolution est tirée par item alors que
        la randomisation de géométrie est par batch ; les deux coexistent.

        `return_z` (ESJEPA) : si True, retourne AUSSI z_pred [B, num_targets,
        z_dim] — les statistiques du résidu prédites par la tête z. Refus
        bruyant si le prédicteur a été construit sans error_signal (un z
        silencieusement absent, c'est un arm qui croit moduler ses quantiles
        et ne module rien). Flag off : signature et retour inchangés.

        Args:
            context_embeddings: Context [B, N_context, d_model]
            num_targets: Number of targets to predict
            attention_mask: Optional mask

        Returns:
            Predictions [B, num_targets, d_model]
            (ou le tuple (predictions, z_pred) si return_z=True)
        """
        if return_z and not hasattr(self, 'z_head'):
            raise ValueError(
                "return_z=True mais le prédicteur a été construit sans "
                "error_signal — l'arm ESJEPA exige model.error_signal=true "
                "à la construction."
            )
        batch_size = context_embeddings.shape[0]

        future_queries = self._future_queries(batch_size, num_targets)

        # G9.2 — conditionnement d'échelle par item (w = k2/k1, [B]).
        if w is not None:
            if not hasattr(self, 'w_film'):
                # Refuser plutôt qu'ignorer : un w silencieusement perdu, c'est
                # un arm inter-résolution qui entraîne SANS conditionnement et
                # des chiffres qu'on croit conditionnés.
                if bool((w != 1).any()):
                    raise ValueError(
                        "w != 1 reçu mais le prédicteur a été construit sans "
                        "use_w_film — l'arm inter-résolution exige que le "
                        "modèle soit construit avec cross_resolution=true."
                    )
            else:
                film = self.w_film(
                    torch.log2(w.to(future_queries.dtype)).unsqueeze(-1))
                gamma, beta = film.chunk(2, dim=-1)              # [B, d] chacun
                future_queries = (
                    future_queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1))

        # Concat
        x = torch.cat([context_embeddings, future_queries], dim=1)
        
        # Transform
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.final_norm(x)

        # Extract targets
        trunk_targets = x[:, -num_targets:, :]
        target_predictions = self.prediction_head(trunk_targets)

        if return_z:
            # ESJEPA — z lu sur le tronc partagé (post-final_norm), pas sur la
            # sortie de prediction_head : les deux voies bifurquent ici.
            return target_predictions, self.z_head(trunk_targets)
        return target_predictions


class MLPPredictor(nn.Module):
    """
    Simple MLP-based predictor (alternative to transformer).
    
    Uses a lightweight MLP to predict each target independently.
    Faster but less powerful than TransformerPredictor.
    
    Args:
        d_model: Model dimension
        num_layers: Number of MLP layers (default: 2)
        hidden_dim: Hidden dimension (default: 2 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        hidden_dim = hidden_dim or 2 * d_model
        
        # Build MLP
        layers = []
        for i in range(num_layers):
            in_dim = d_model if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict targets from context.
        
        For MLP, we use mean pooling of context to predict each target.
        
        Args:
            context_embeddings: Context [B, N_context, d_model]
            target_positions: Ignored for MLP (can predict any number)
            
        Returns:
            Predictions [B, N_context, d_model] (same size as input)
        """
        # Mean pool context
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        # [B, 1, d_model]
        
        # Expand to match input size (predict all positions)
        num_positions = context_embeddings.shape[1]
        context_pooled = context_pooled.expand(-1, num_positions, -1)
        
        # MLP prediction
        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions
    
    def forward_simple(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int,
        w=None,
        **kwargs
    ) -> torch.Tensor:
        """Simple forward for N targets."""
        # Le MLP mean-poole le contexte : il n'a ni ordre ni requêtes, donc
        # aucun endroit où un conditionnement d'échelle aurait du sens. Sans
        # cette garde, **kwargs avalait `w` en silence — l'arm inter-résolution
        # aurait « tourné » sans conditionnement.
        if w is not None and bool((w != 1).any()):
            raise NotImplementedError(
                "MLPPredictor ne supporte pas le conditionnement w (G9.2) — "
                "utiliser predictor_type='transformer'."
            )
        # Même famille de garde pour ESJEPA : **kwargs ne doit pas avaler
        # return_z en silence (un z jamais produit = quantiles jamais modulés).
        if kwargs.get('return_z', False):
            raise NotImplementedError(
                "MLPPredictor ne supporte pas la voie z (ESJEPA) — "
                "utiliser predictor_type='transformer'."
            )
        # Mean pool
        context_pooled = context_embeddings.mean(dim=1, keepdim=True)
        context_pooled = context_pooled.expand(-1, num_targets, -1)
        
        # Predict
        predictions = self.mlp(context_pooled)
        predictions = self.norm(predictions)
        
        return predictions