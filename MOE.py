import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.fc = nn.Linear(in_features,out_features)

    def forword(self,x):
        return self.fc(x)
    
class MoELayer(nn.Module):
    def __init__(self,in_features,out_features,num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features,out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features,num_experts)
    
    def forward(self,x):
        gate_score = F.softmax(self.gate(x),dim=-1)
        expert_outputs=torch.stack([expert(x) for expert in self.experts])
        output=torch.bmm(gate_score.unsqueeze(1),expert_outputs).squeeze(1)
        return output
    
input_size=5
output_size=3
num_experts=4
batch_size=10

model =MoELayer(input_size, output_size, num_experts)
demo=torch.randn(10,input_size)
output=model(demo)
print(output.shape) #(10,3)