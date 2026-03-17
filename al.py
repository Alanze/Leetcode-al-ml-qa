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
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # 合并头部并线性变换输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        vals = []
        node = self
        while node:
            vals.append(str(node.val))
            node = node.next
        return "->".join(vals)
    
    @staticmethod
    # 删除链表中的重复的节点
    def delete_node(head):
        if not head:
            return None
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head

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
    print("Multi-Head Attention Output Shape:", output.shape)
    # 测试链表节点删除
    head = ListNode(1, ListNode(2, ListNode(2, ListNode(2, ListNode(3)))))
    print("Original List:", head)
    new_head = ListNode.delete_node(head)
    print("List after deleting duplicates:", new_head)
