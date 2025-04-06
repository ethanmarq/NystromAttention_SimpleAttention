import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""
Nystrom Code
"""
class NystromSelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_landmarks, mask=False, CUDA=False, conv_kernel_size=None):
        super(NystromSelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.num_landmarks = num_landmarks
        self.dropout = nn.Dropout(0.1)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")
        self.init_option = "original"

        # Currently trails does not use convolution
        self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,  
                out_channels=1,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False
            ).to(self.device)

    def forward(self, query_in, key_in, value_in):
        batch_size = query_in.size(0)
        # Query Sequence Length
        q_seq_len = query_in.size(1)  
        # Key Sequence length
        k_seq_len = key_in.size(1)    
        
        # Linear projections
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        
        # Scale for stability
        scaling = 1.0 / math.sqrt(math.sqrt(self.d_k))
        query = query * scaling
        key = key * scaling
        
        # Create proper causal mask if needed
        if self.mask:
            mask_tensor = torch.ones(batch_size, q_seq_len, k_seq_len, device=self.device)
            mask_tensor = torch.tril(mask_tensor)  # Lower triangular mask

        # Case for decoder self-attention (q_seq_len = 1, k_seq_len > 1)
        is_decoder_self_attn = (q_seq_len == 1 and k_seq_len > 1)
        
        # Optional Nystrom Attention which may improve performance 
        # for low squenece length cases
        #use_nystrom = seq_len > 4 and seq_len > self.num_landmarks
        
        # Current training forces Nystrom to always be used
        use_nystrom = True #
        
        if use_nystrom:
            # Find the largest divisor of seq_len that's <= num_landmarks
            # This prevents errors
            num_landmarks = self.num_landmarks
            while k_seq_len % num_landmarks != 0 and num_landmarks > 1:
                num_landmarks -= 1
            
            segments = k_seq_len // num_landmarks
            
            # Fallback case for saftey
            if num_landmarks == 0:
                # Fall back to standard attention for simplicity
                use_nystrom = False
                print(f"  Falling back to standard attention (no good divisor found)")
            else:
                #print(f"  Using {num_landmarks} landmarks with {segments} segments")
                pass

        # Standard Attention
        if not use_nystrom:
            print(f"Using standard attention with seq_len={seq_len}")
            # Standard attention implementation
            key_transposed = torch.transpose(key, 1, 2)
            attention_weights = torch.matmul(query, key_transposed)
            
            if self.mask:
                # Apply mask
                attention_weights = attention_weights.masked_fill(
                    mask_tensor.unsqueeze(1) == 0, float('-inf')
                )
            
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weighted_value = torch.matmul(attention_weights, value)
        else:
            # Nystrom attention implementation
            
            # 1. Create landmark tokens
            try:
                # Only reshape key and value, not query if it's decoder self-attention
                key_reshaped = key.reshape(batch_size, num_landmarks, segments, -1)
                
                # Create landmarks by averaging each segment
                k_landmarks = key_reshaped.mean(dim=2)    # [batch_size, num_landmarks, d_k]

                # With this decoder this statement will always evaluate true
                if is_decoder_self_attn:
                    q_landmarks = query
                else:
                    # Normal case, reshape query
                    query_reshaped = query.reshape(batch_size, num_landmarks, segments, -1)
                    q_landmarks = query_reshaped.mean(dim=2)
                
                # 2. Compute the three Nystrom kernels
                # kernel_1: [batch_size, seq_len, num_landmarks]
                kernel_1 = torch.matmul(query, k_landmarks.transpose(-1, -2))

                kernel_1 = F.softmax(kernel_1, dim=-1)
                
                # kernel_2: [batch_size, num_landmarks, num_landmarks]
                kernel_2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))

                kernel_2 = F.softmax(kernel_2, dim=-1)
                
                # kernel_3: [batch_size, num_landmarks, seq_len]
                kernel_3 = torch.matmul(q_landmarks, key.transpose(-1, -2))

                
                if self.mask:
                    if is_decoder_self_attn:
                        seq_mask = torch.ones(batch_size, 1, k_seq_len, device=self.device)
        
                    last_pos = q_seq_len - 1
                    for i in range(k_seq_len):
                        if i > last_pos:
                            seq_mask[:, :, i] = 0
                    else:
                        # For regular causal masking
                        seq_mask = torch.tril(torch.ones(batch_size, num_landmarks, k_seq_len, device=self.device))
                    # Apply mask
                    kernel_3 = kernel_3.masked_fill(seq_mask == 0, float('-inf'))
                
                kernel_3 = F.softmax(kernel_3, dim=-1)
                
                # 3. Compute Nystrom approximation
                # Moore-Penrose pseudoinverse of kernel_2
                kernel_2_inv = torch.pinverse(kernel_2)
                #kernel_2_inv = self.iterative_inv(kernel_2) # Debugging as there is a shape mismatch
                
                # Final Nystrom approximation
                attention_weighted_value = torch.matmul(
                    torch.matmul(kernel_1, kernel_2_inv),
                    torch.matmul(kernel_3, value)
                )
                
            except RuntimeError as e:
                print(f"Error in Nystrom attention: {e}")
                print(f"Falling back to standard attention due to error")
                
                # Fall back to standard attention on error
                key_transposed = torch.transpose(key, 1, 2)
                attention_weights = torch.matmul(query, key_transposed)
                
                if self.mask:
                    attention_weights = attention_weights.masked_fill(
                        mask_tensor.unsqueeze(1) == 0, float('-inf')
                    )
                
                attention_weights = F.softmax(attention_weights, dim=-1)
                attention_weighted_value = torch.matmul(attention_weights, value)
        
        # Convolution if conv_kernel_size is not none
        if self.use_conv:
            v_conv = value.unsqueeze(1)
            v_conv = self.conv(v_conv)
            
        attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value
        
    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
    
        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)
    
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

class NystromMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False):
        super(NystromMultiHeadAttention, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [NystromSelfAttention(embed_dim, d_k, d_v, num_landmarks, mask, conv_kernel_size, CUDA) for _ in range(num_heads)]
        )
        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat(
                (attention_out, attention(query, key, value)), dim=2
            )
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm

class NystromTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False):
        super(NystromTransformerBlock, self).__init__()
        self.multi_head_attention = NystromMultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            num_landmarks,
            mask,
            conv_kernel_size,
            CUDA=CUDA,
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim)

    def forward(self, query, key, value, residual_x):
        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out

"""
Non-Nystrom Code
"""
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, mask=False):
        super(SelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_in, key_in, value_in):
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        key_transposed = torch.transpose(key, 1, 2)
        # Get attention weights
        attention_weights = torch.matmul(query, key_transposed)  # (n_query,n_key)
        attention_weights = attention_weights / math.sqrt(self.d_k)
        if self.mask == True:
            # REF : http://peterbloem.nl/blog/transformers
            indices = torch.triu_indices(
                attention_weights.shape[1], attention_weights.shape[2], offset=1
            )
            attention_weights[:, indices[0], indices[1]] = float("-inf")
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention weights to value
        attention_weighted_value = torch.matmul(
            attention_weights, value
        )  # (n_query,n_key) matmul (n_key || n_query , d_v)
        attention_weighted_value = self.dropout(attention_weighted_value)

        return attention_weighted_value


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, mask=False, CUDA=False):
        super(MultiHeadAttention, self).__init__()
        ### Credit: Issue From @shouldsee https://github.com/IpsumDominum/Pytorch-Simple-Transformer/issues/2
        self.attention_blocks = nn.ModuleList(
            [SelfAttention(embed_dim, d_k, d_v, mask) for _ in range(num_heads)]
        )

        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat(
                (attention_out, attention(query, key, value)), dim=2
            )
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm


class LayerNorm(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        div = (std + self.eps) + self.shift
        return self.scale * (x - mean) / (div)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(embed_dim, output_dim)
        self.RELU = nn.ReLU()
        self.l2 = nn.Linear(output_dim, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, residual_x):
        x = torch.max(torch.zeros(x.shape), self.l1(x))
        x = self.RELU(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=False, CUDA=False):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            mask,
            CUDA=CUDA,
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim)

    def forward(self, query, key, value, residual_x):
        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out


class VocabLogits(nn.Module):
    def __init__(self, embed_dim, logit_dim):
        super(VocabLogits, self).__init__()
        self.linear = nn.Linear(embed_dim, logit_dim)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class Embeddings(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, vocab_length, embed_dim, CUDA=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_length, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, CUDA=CUDA)
        self.embed_dim = embed_dim

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.embed_dim)
        return embed + self.pos_encode(embed)


class PositionalEncoding(nn.Module):
    "Modified From Annotated Transformer (HarvardNLP)"

    def __init__(self, embed_dim, max_len=5000, CUDA=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(
            10000.0, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        div_term_odd = torch.pow(
            10000.0, torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)
        if CUDA == True:
            pe.type(torch.cuda.FloatTensor)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return x
