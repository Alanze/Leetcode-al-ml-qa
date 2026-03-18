# 手写transformer结构
import torch
import torch.nn as nn   
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 线性变换并分头
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        print("After linear transformation and splitting heads:")
        print("Query shape:", query.shape)
        print("Key shape:", key.shape)
        print("Value shape:", value.shape)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        print("Attention scores shape:", scores.shape)
        print("Sample scores (first head, first batch):", scores[0, 0, :3, :3])
        
        attn_weights = torch.softmax(scores, dim=-1)
        print("Attention weights shape:", attn_weights.shape)
        print("Sample attention weights (first head, first batch):", attn_weights[0, 0, :3, :3])
        
        attn_output = torch.matmul(attn_weights, value)
        print("Attention output shape:", attn_output.shape)
        print("Sample attention output (first head, first batch):", attn_output[0, 0, :3, :10])
        
        # 合并头部并线性变换输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        print("After concatenating heads shape:", attn_output.shape)
        
        output = self.out_linear(attn_output)
        print("Final output shape:", output.shape)
        
        return output

if __name__ == "__main__":
    # 测试多头注意力机制
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_length = 10
    
    attention = MultiHeadAttention(d_model, num_heads)
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)
    
    output = attention(query, key, value)
    print(output)
    print("Multi-Head Attention Output Shape:", output.shape)

