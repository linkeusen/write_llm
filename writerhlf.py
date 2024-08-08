import torch
import random
from transformers import AutoTokenizer,default_data_collator,AutoModelForCausalLM,get_scheduler,AutoModel
from accelerate import Accelerator
from datasets import load_dataset


# 对一个线性层进行Lora操作
 #首先定义一个Lora类，继承torch.nn.Module  
class Lora(torch.nn.Module):
    def __init__(self,linear):
        super().__init__()
        self.linear = linear
        self.Lora_A=torch.nn.Parameter(torch.randn(linear.in_features,128)*0.1)
        self.Lora_B=torch.nn.Parameter(torch.zeros(128,linear.out_features))
        self.linear.weight.requires_grad = False
    def forward(self, x):
        y_linear=self.linear(x)
        y_lora=x.matmul(self.Lora_A).matmul(self.Lora_B)
        return y_linear+y_lora/128
 # 修改线性层
def insert(model):
    for name,layer in model.named_modules():
        if '这里应该为解码层的名字' not in name:
            continue
        if not isinstance(layer,torch.nn.Linear):
            continue
        name=name.split('.')
        for i in name[:-1]:
            layer_father=getattr(model,i)
        lora_layer=Lora(layer)
        layer_father.__setattr__(name[-1],lora_layer)
 
 #合并lora层到原始的线性层权重
def merge(model):
    for name,layer in model.named_modules():
        if not isinstance(layer,Lora):
            continue
        linear=layer.linear
        # 是否要.t()？
        linear.weight.data+=layer.Lora_A.matmul(layer.Lora_B).t()/128
        name=name.split('.')
        for i in name[:-1]:
            layer_father=getattr(model,i)
        layer_father.__setattr__(name[-1],linear)  

# 定义一个tokenizer类
class Tokenizer():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
    def encode(self,input,max_length=512):
        input_ids=self.tokenizer.encode(input,add_special_tokens=False)
        if len(input_ids)>max_length-2:
            input_ids=[self.bos_token_id]+input_ids[:max_length-2]+[self.eos_token_id]
        else:
            input_ids=[self.bos_token_id]+input_ids+[self.pad_token_id]*(max_length-len(input_ids)-2)+[self.eos_token_id]
        input_ids=torch.LongTensor(input_ids)
        attention_mask=(input_ids!=self.pad_token_id).long()
        return input_ids,attention_mask   
    def decode(self,input_ids):
        input_ids=input_ids.tolist()
        if self.eos_token_id in input_ids:
            end=input_ids.index(self.eos_token_id)+1
            input_ids=input_ids[:end]
        return self.tokenizer.decode(input_ids)
    def pad_to_left(self,input_ids):
        input_ids=input_ids.tolist()
        end=input_ids.index(self.eos_token_id)
        input_ids[end]=self.pad_token_id
        input_ids=input_ids[end:]+input_ids[:end]
        input_ids=torch.LongTensor(input_ids)
        attention_mask=(input_ids!=self.pad_token_id).long()
        return input_ids,attention_mask
    
tokenizer=Tokenizer()

###################################################################################################
# 加载actor数据集
dataset=load_dataset('json',data_files='..',split='train')
dataset_actor=dataset.select(range(15000))
# 数据集批处理
def f(data):
    if random.random()>0.5:
        data['chosen']=data['chosen'].swapcase()
    data=data['prompt']+data['chosen']
    input_ids,attention_mask=tokenizer.encode(data)
    return {'input_ids':input_ids,'attention_mask':attention_mask}
dataset_actor=dataset_actor.map(f,remove_columns=dataset_actor.column_names)
# 加载为loader
loader_actor=torch.utils.data.DataLoader(dataset_actor,collate_fn=default_data_collator,batch_size=2,shuffle=2,drop_last=True)
# 加载模型
model_actor=AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b')
Lora.insert(model_actor)
# 加载训练组件
param=[]
param_lora=[]
def f1():
    for name,para in model_actor.named_parameters():
        if not para.requires_grad:
            continue
        if 'Lora_A' or 'Lora_B' in name:
            param_lora.append(para)
            continue
        param.append(para)
    return [{'params':param,'weight_decay':0.0,'lr':5e-4},{'params':param_lora,'weight_decay':0.0,'lr':5e-4}]
 # 加载优化器和学习率调度器
optimizer=torch.optim.Adam(f1(),lr=1e-3,betas=(0.9,0.98),eps=1e-8)
scheduler = get_scheduler(name='cosine',optimizer=optimizer,num_warmup_steps=0,num_train_steps=100)
 # 加速器
accelerator=Accelerator(gradient_accumulation_steps=64,
                          mixed_precision='fp16')
 # 用accelerator合并 
model_actor,optimizer,loader_actor,scheduler=accelerator.prepare(model_actor,optimizer,loader_actor,scheduler)
 # 训练模型
model_actor.train()
for data in loader_actor:
    # 梯度累积
    with accelerator.accumulate(model_actor):
        out=model_actor(**data)
        # 反向传播损失
        accelerator.backward(out.loss)
        # 是否到到梯度累积步数
        if accelerator.sync_gradients:
            # 裁剪梯度使最大值小于1
            accelerator.clip_grad_norm_(
                [p for p in model_actor.parameters() if p.required_grad],1.0
            )
        # 更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
# 合并lora层并保存模型
Lora.merge(model_actor)
model_actor.save_pretrained('..')

# ######################################################################

# 加载crtic数据集
dataset_critic=dataset.select(range(15000,30000))
# 分别将chosen和critic与prompt合并，然后tokenizer化后再合并
def f2(data):
    chosen=data['prompt']+data['chosen'].swapcase()
    rejected=data['prompt']+data['chosen']
    input_ids_chosen,attention_mask_chosen=tokenizer.encode(chosen)
    input_ids_rejected,attention_mask_rejected=tokenizer.encode(rejected)
    return {'input_ids_chosen':input_ids_chosen,'attention_mask_chosen':attention_mask_chosen,'input_ids_rejected':input_ids_rejected,'attention_mask_rejected':attention_mask_rejected}
dataset_critic=dataset_critic.map(f2,remove_columns=dataset_critic.column_names)
dataset_critic=dataset_critic.set_format('torch')

def f3(data):
    chosen_input_ids=[i['input_ids_chosen'] for i in data]
    chosen_attention_mask=[i['attention_mask_chosen'] for i in data]
    rejected_input_ids=[i['input_ids_rejected'] for i in data]
    rejected_attention_mask=[i['attention_mask_rejected'] for i in data]
    input_ids=torch.stack(chosen_input_ids+rejected_input_ids,dim=0)
    attention_mask=torch.stack(chosen_attention_mask+rejected_attention_mask,dim=0)
    return {'input_ids':input_ids,'attention_mask':attention_mask}
# 加载为loader
loader_critic=torch.utils.data.DataLoader(dataset_critic,collate_fn=f3,batch_size=2,shuffle=2,drop_last=True)
# 定义critic模型类，在原始的预训练模型后添加一个线性层
class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rwmodel=AutoModel.from_pretrained('facebook/opt-1.3b')
        self.v_head=torch.nn.Linear(512,1,bias=False)
    def forward(self,input_ids,attention_mask):
        out=self.rwmodel(input_ids,attention_mask).last_hidden_state
        out=self.v_head(out).squeeze(-1)
        # 计算损失,此处有点疑问
        loss_sum=0.0
        value_chosen_sum=0.0
        value_rejected_sum=0.0
        for input_ids_chosen,input_ids_rejected,value_chosen,value_rejected in zip(
            input_ids[:4],input_ids[4:],
            out[:4], out[4:]
        ):
            start = (
                input_ids_chosen == input_ids_rejected).tolist().index(False)

            end_chosen = input_ids_chosen.tolist().index(
                tokenizer.eos_token_id) + 1
            end_rejected = input_ids_rejected.tolist().index(
                tokenizer.eos_token_id) + 1
            end = max(end_chosen, end_rejected)

            value_chosen = value_chosen[start:end]
            value_rejected = value_rejected[start:end]

            loss = value_chosen - value_rejected
            loss = -torch.nn.functional.logsigmoid(loss).mean()

            loss_sum += loss
            value_chosen_sum += value_chosen.mean().item()
            value_rejected_sum += value_rejected.mean().item()
      
        return loss_sum / 4, value_chosen_sum, value_rejected_sum  

# 加载模型进入训练模型
model_critic=Critic()
def f4():
    params_decay=[]
    params=[]
    for name,para in model_critic.named_parameters():
        if 'bios' or 'norm.weight' not in name:
            params.append(para)
            continue
        params_decay.append(para)
    return[
        {'params':params,'weight_decay':0.0},
        {'params':params_decay,'weight_decay':0.1}
    ]
optimizer_critic=torch.nn.Adam(f4(),lr=5e-3,betas=(0.9,0.98),eps=1e-8)
scheduler_critic=get_scheduler(name='cosine',optimizer=optimizer_critic,num_warmup_steps=0,num_train_steps=100)
accelerator=Accelerator(gradient_accumulation_steps=64,mixed_precision='fp16')
model_critic,optimizer_critic,loader_critic,scheduler_critic=accelerator.prepare(model_critic,optimizer_critic,loader_critic,scheduler_critic)
model_critic.train()
# 开始训练模型
for data in loader_critic:
    with accelerator.accumulate(model_critic):
        loss,value_chosen_sum,value_rejected_sum=model_critic(**data)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                [p for p in model_critic.parameters() if p.requires_grad],1.0
            )
        optimizer_critic.step()
        scheduler_critic.step()
        optimizer_critic.zero_grad()
# 保存模型
# model_critic.save_pretrained('...')
# 如果只保存模型配置model_critic.state_dict()
torch.save(model_critic.to('cpu'),'...')

###################################################################################################

# 加载rlhf所需数据
dataset_rlhf=dataset.select(range(45000,len(dataset)))
# 转换pad到左边
def f5(data):
    input_ids,_=tokenizer.encode(data['prompt'],max_length=256)
    input_ids,attention_mask=tokenizer.pad_to_left(input_ids)
    return {'input_ids':input_ids, 'attention_mask':attention_mask}
# 处理数据集
dataset_rlhf=dataset_rlhf.map(f5,remove_columns=dataset_rlhf.column_names)
# 加载loader
loader_rlhf=torch.utils.data.DataLoader(dataset_rlhf,collate_fn=default_data_collator,batch_size=4,shuffle=True,drop_last=True)


