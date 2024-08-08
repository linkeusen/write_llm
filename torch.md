# torch.
ones(2,3)
zeros(2,3)
eye(3,3)---主对角为1的对角矩阵
full([3,3],7)
empty(2,3)
randn(2,3)---均值为0方差为1标准正态分布
rand(2,3)----01均匀分布
randint(2,6,[2,3])----2到6的整数均匀分布
arange(1,10,2)----步长为2自增
linspace(1,10,2)---等差
clamp(最小值，最大值)---区间
## 数据类型
FloatTensor([1,2])--浮点类型tensor
LongTensor([1,2])---整数类型tensor
## 维度变化
reshape()---变换
view()---变换
unsqueeze(-1)---插入维度的位置
squeeze(-1)---删除维度
repeat(2,3)---第一个维度复制两次第二个维度复制3次
transpose(0,1)---只能两两交换
permute(2,1,0)---交换维度顺序
t()---2维转置
## 广播
expand_as()
## 拼接和拆分
.cat([a,b],dim=0)---在指定维度上拼接a和b
.stack([a,b]，dim=0)---新加一个0维度上拼接ab
.spilt(2,dim=0)---0维上每2个拆一次
.spilt([1,3,2],dim=0)---在0维上拆分为1,3,2
.chunk(2,dim=0)---在0维上拆分为2个
## 数学计算
a@b---
a.matmul(b)---广播矩阵乘法
**2---求指数
**0.5---开根号
.exp()---e的n次方
.log()---以e为底求对数
.log2()---以2为底求对数
.clamp(2,4)---裁剪限制数据上下限
.floor()---向下取整
.ceil()---向上取整
.round()---四舍五入
## 装饰函数
@property---通过.像调用属性一样调用函数
@torch.inference_mode()---执行时自动切换到推理模式
@staticmethod---不需要通过类的实例来调用
## 属性统计
.max(dim=0)---返回指定维度上的最大值和索引
.argmax(dim=0)---只返回最大值的索引
.argmin()
.max()
.min()
.mean()
.prod()---求积
.sum()
.norm(1,dim=0)---求1范数
.topk(2,dim=1,largest=False)---求前两个最小值
.kthvalue(2,dim=1)---求第二个小值
# 初始化
kaiming初始化
torch.nn.init.kaiming_normal_(cov.weight.data)
常数初始化
torch.nn.init.constant_(linear.weight.data)

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return len(self)
    def __getitem__(self,i)
# 定义loader
loader=torch.utils.data.Dataloader(
    dataset-dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True
)
# 神经网络
torch.nn.Sequential(
    torch.nn.linear(),
    ...
)---直接定义多个层

# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr=le-4)
loss_func=torch.nn.CrossEntropyLoss()
loss=loss_func(out,y)
loss.backward()
optimizer.strp()
optimizer.zero_grad()
# 保存torch模型加载
保存模型
torch.save(model,dir)
torch.load(dir)
保存权重
torch.save(model.state_dict,dir)
torch.load_state_dict(dir)
# huggingface
## load_from_disk
from datasets import load_from_disk
dataset= load_from_disk('...')

del dataset['...']---删除字段
dataset.map(function，batched=True,batch_size=5,num_proc=2线程)---数据集上遍历应用函数
dataset.filter(function，)---过滤数据集
dataset.remove_columns(['...'])---删除字段
dataset.rename_columns(['...'])---重命名
dataset.set_format('pt',column=['...'],output_all_columns=True)---定义数据类型
concatenate_datasets(list(dataset.values()))---合并多个数据集
dataset['...'].train_test_spilt(test_size=0.1,train_size=8000)---切分数据集
dataset['...'].select([5,15,20,30])---取数据子集
## Dataset
from datasets import Dataset
dataset=一个字典
dataset=Dataset.from_dict(dataset)---从字典创建数据集
dataset=Dataset.from_generator(function)---从函数生成数据集
## load_dataset
from datasets import load_dataset
dataset=load_dataset('csv',data_files='../.csv',split='train')---从csv创建数据集
dataset.to_csv('../.csv')---保存一个数据集为csv
dataset.to_json('../.json')---
## 创建模型
from transforms import AutoModel
model = AutoModel.from_pretrained('...')

model.save_pretrained('...')
 

# 命令行
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--..',type=,default=,help='')
args=parser.parse_args()

# fastapi
from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def readroot():
    return {'hallo fuking world'}

## uvicorn启动默认127.0.0.1:8000端口，命令窗口
uvicorn 程序名:实例名 --reload自动重载 --port 8080端口 --host 0.0.0.0 ip地址
curl -X 选项来更改默认的get这个行为以发送 POST、PUT、DELETE、HEAD 
?变量=变量值传入参数
curl -X POST "http://127.0.0.1:8000/items/" -H"accept:application/json" -H"Content-Type:application/json" -d"{\"name\":\"test\",\"price\":12.5}"


# git
## 已有仓库下首次推送main
一：git clone 已经创建的.git
二：cd到当前目录
三：git add. 跟踪所有文件
四：git commit -m "创建一个包含这些更改的新提交"
五：git push origin main 推送到main分支，如果为第一次
git push -u origin main 设置为上游分支

