import torch
import torch.nn as nn

class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim, attn_dim=256):
        super().__init__()
        # Nonlinear transform for score computation (equivalent to layer 1)
        self.V = nn.Linear(hidden_dim, attn_dim)
        # Projection to a score (equivalent to layer 2)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        # h: (L, 512)
        v = torch.tanh(self.V(h)) 
        a = self.w(v).squeeze(-1)
        alpha = torch.softmax(a, dim=0)
        
        # Aggregate via weighted sum
        z = (alpha.unsqueeze(-1) * h).sum(dim=0)
        return z, alpha

class GatedAttentionPooler(nn.Module):
    def __init__(self, hidden_dim, attn_dim=256):
        super().__init__()
        # Layer-1 equivalent: two different nonlinear transforms
        self.V = nn.Linear(hidden_dim, attn_dim) # Extract salient features
        self.U = nn.Linear(hidden_dim, attn_dim) # Gate/filter pathway
        
        # Layer-2 equivalent: projection to score
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        # h: (L, 512)
        v_tanh = torch.tanh(self.V(h))
        v_sigm = torch.sigmoid(self.U(h))
        
        # Multiply the two outputs (gated mechanism)
        v = v_tanh * v_sigm # (L, 256)
        
        a = self.w(v).squeeze(-1) # (L,)
        alpha = torch.softmax(a, dim=0)
        
        z = (alpha.unsqueeze(-1) * h).sum(dim=0)
        return z, alpha

class Detection_Model(nn.Module):
    def __init__(self, n_features=4096, reduced_dim=256,pooling_method='max', p_norm=10.0, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pooling_method = pooling_method
        self.p_norm = p_norm

        use_attention_features = pooling_method in {"attention", "gated_attention"}
        self.feature_dim = n_features if use_attention_features else reduced_dim

        if use_attention_features:
            self.feature_extractor = None
        else:
            # Feature extraction layer (compress only for non-attention pooling)
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, reduced_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ).to(self.device)

        attn_hidden_dim = max(min(256, self.feature_dim), 1)

        self.bag_classifier = nn.Linear(self.feature_dim, 1).to(self.device)
        self.attention_pooler = AttentionPooler(self.feature_dim, attn_dim=attn_hidden_dim).to(self.device)
        self.gated_attention_pooler = GatedAttentionPooler(self.feature_dim, attn_dim=attn_hidden_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3]).to(self.device))
    def forward(self, hidden_states, labels=None):
        bag_logits = []
        bag_scores = []

        for h in hidden_states:
            h = h.to(self.device).float()
            h_layer = h[:, 0, :]  # (L, 4096)

            # For attention-based pooling, use original features without compression
            if self.feature_extractor is not None:
                h_reduced = self.feature_extractor(h_layer)  # (L, D)
            else:
                h_reduced = h_layer

            if self.pooling_method == 'max':
                z = torch.max(h_reduced, dim=0)[0]
            elif self.pooling_method == 'mean':
                z = torch.mean(h_reduced, dim=0)
            elif self.pooling_method == 'attention':
                z, _ = self.attention_pooler(h_reduced)
            elif self.pooling_method == 'gated_attention':
                z, _ = self.gated_attention_pooler(h_reduced)
            elif self.pooling_method == 'lp':
                z = torch.norm(h_reduced, p=self.p_norm, dim=0)

            logit = self.bag_classifier(z.unsqueeze(0))   # (1,1)
            score = torch.sigmoid(logit)

            bag_logits.append(logit)
            bag_scores.append(score)

        all_logits = torch.cat(bag_logits).flatten()
        all_scores = torch.cat(bag_scores).flatten()

        return {
            "loss": None if labels is None else self.criterion(all_logits, labels.float()),
            "logits": all_logits,
            "scores": all_scores,
            "attentions": None,
            "bag_representations": z.unsqueeze(0)
        }
