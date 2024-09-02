#首先根据hf网站命令配置accelerate config或者输入accelerate config进行交互式配置

#数据并行
from accelerate import Accelerator,DeepSpeedPlugin
import torch
from torch.utils.data import TensorDataset, DataLoader

class SimpleNet(torch.nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        super().__init__()
        self.fc1= torch.nn.Linear(input_dim,hidden_dim)
        self.fc2= torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

if __name__ == "__main__":
    input_dim=10
    output_dim=5    
    hidden_dim=20
    batch_size=64
    data_size=10000

    input_data=torch.randn(data_size, input_dim)
    labels=torch.randn(data_size, output_dim)

    dataset=TensorDataset(input_data, labels)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model =SimpleNet(input_data,hidden_dim,output_dim)
# zero_stage
    deepspeed=DeepSpeedPlugin(zero_stage=2,gradient_clipping=1.0)
    accelerator = Accelerator(deepspeed_plugin=deepspeed)
    optimizator = torch.optim.Adam(model.parameters(),lr=0.001)
    crition =torch.nn.MSELoss()
# 只放入与模型相关的参数
    model,optimizator,dataloader = accelerator(model,optimizator,dataloader)

    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs,labels= batch
            optimizator.zero_grad()
            outputs=model(inputs)
            loss =crition(outputs,labels)
            accelerator.backward(loss)
            optimizator.step()
        print(f"epoch:{epoch},loss:{loss.item()}")

    accelerator.save(model.state_dict(), "model.pth")

    #运行
    #python -m accelerate.commands.launch xxx.py
    