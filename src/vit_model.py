from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(self, 
                 
                 hidden_size = 1152,
                 intermediate_size = 4304,
                 num_hidden_layers = 27,
                 num_attention_heads = 16,
                 num_channels = 3,
                 image_size = 224,
                 patch_size = 14,
                 layer_norm_eps = 1e-6,
                 attention_dropout= 0.0,
                 num_image_tokens = None,
                 **kwargs
                ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens if num_image_tokens else (image_size // patch_size) ** 2
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.embd_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding =  nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding= "valid"
        )
        
        self.position_embedding = nn.Embedding(config.num_image_tokens, self.embd_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_image_tokens).expand((1, -1)),
            persistent=False,
        )
        
    def forward(self, pixel_values : torch.FloatTensor) -> torch.Tensor:
               
        patch_embd = self.patch_embedding(pixel_values)
        patch_embd = patch_embd.flatten(2)
        patch_embd = patch_embd.transpose(1, 2)
        embeddings = patch_embd + self.position_embedding(self.position_ids)
        return embeddings
    
    
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.embd_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embd_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.v_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_state.size()
        
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        
        query_state = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_state = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_state = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # calculate the attention score
        attn_scores = torch.matmul(query_state, key_state.transpose(2,3))*self.scale
        
        if attn_scores.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_scores.size()}"
            )
            
        # Apply the softmax row-wise
        attn_scores = nn.functional.softmax(attn_scores, dim=-1, dtype= torch.float32).to(query_state.dtype)
        attn_scores = nn.functional.dropout(attn_scores, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_scores, value_state)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embd_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.embd_dim = config.hidden_size
        self.self_attn = SiglipAttention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config=config)
        self.layer_norm2 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        hidden_states = self.self_attn(self.layer_norm1(hidden_states)) + hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states)) + hidden_states
        return hidden_states
    
class SiglipEncoder(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config=SiglipVisionConfig):
        super().__init__()
        
        self.embd_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states
    
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.vision_model = SiglipVisionTransformer(config=config)
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, num_image_tokens, Embed_Dim]
        return self.vision_model(pixel_values)