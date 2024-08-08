import math 
import torch

# 均方根RMSnorm层
class LlamaRMSNorm(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.weight=torch.nn.Parameter(torch.ones(1024))
      return
   def forward(self, x):
    var=x.pow(2).mean(2,keepdim=True)
    x=x*(var+1e-5).rsqrt()
    return self.weight*x
   
# 位置编码
@torch.no_grad()
def llama_rotary_embedding(len):
   inv_freq=torch.arange(0,len*2,2)/(len*2)
   inv_freq=1.0/(50000**inv_freq)
   inv_freq.reshape(1,len,1)
   position_ids=torch.arange(len).reshape(1,1,-1)
   freqs=inv_freq.matmul(position_ids).transpose(1,2)
   emb=torch.cat((freqs,freqs),2)
   return emb.cos(),emb.sin()

# 将位置信息编码到输入张量中
def apply_rotary_pos_emb(x,cos,sin)
    def rotate_half(x):
        left=x[...,:x.shape[-1]//2]
        right=x[...,x.shape[-1]//2:]
        return torch.cat((-right,left),-1)
    cos=cos.unsqueeze(1)
    sin=sin.unsqueeze(1)
    x=(x*cos)+(rotate_half(x)*sin)
    return x

# 简单线性层
class LlamaNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=torch.nn.Linear(1024,14336,bias=False)
        self.silu=torch.nn.SiLU()
        self.up=torch.nn.Linear(1024,14336,bias=False)
        self.down=torch.nn.Linear(14336,1024,bias=False)
        return
    def forward(self, x):
        left=self.silu(self.gate(x))
        right=self.up(x)
        return self.down(left*right)
    
# 实现多头注意力中的kv重复
def repeat_kv(x):
   shape=list(x.shape)
   shape[1]*=4
   return x.unsqueeze(2).repeat(1,1,4,1,1).reshape(shape)

# 通过attention_mask获得遮罩
def get_causal_mask(attention_mask):
   b,len=attention_mask.shape
   min_value=1e-5
   causal_mask=torch.full((len,len),1e-5).triu(diagonal=1)
   causal_mask=causal_mask.reshape(1,1,len,len).repeat(b,1,1,1)
   mask=attention_mask.reshape(b,1,1,len)==0
   causal_mask=causal_mask.masked_fill(mask,min_value)
   return causal_mask

# 注意力层
class LlamaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        return
    def forward(self, x):
        
        return 

# Decoder层
class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return
    
# 输出隐藏张量的通用模型，没有特定的预测目标
class LlamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_ids,attention_mask):
        return hidden_states
    

# 输出概率分布的序列预测模型，输出表示下一个单词的概率分布
class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input_ids,attenuation_mask,labels=None):
        return loss,logits