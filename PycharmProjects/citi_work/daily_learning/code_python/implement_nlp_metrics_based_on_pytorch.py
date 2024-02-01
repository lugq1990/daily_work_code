import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 模型定义
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

# 创建训练数据加载器
def create_data_loader(batch_size=64):
    # 假设有一个简单的任务，输入为序列，输出为标签
    seq_length = 10
    vocab_size = 1000
    num_classes = 5

    # 创建随机训练数据
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, num_classes, (batch_size,))

    # 将数据包装成 TensorDataset
    dataset = TensorDataset(inputs, labels)

    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# 计算困惑度的函数
def compute_perplexity(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            hidden = torch.zeros(1, inputs.size(0), hidden_size)
            inputs, labels = inputs.to(device), labels.to(device)

            logits, _ = model(inputs, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    perplexity = torch.exp(total_loss / total_samples)
    return perplexity

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
input_size = 1000
hidden_size = 128
output_size = 5
model = SimpleModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建训练数据加载器
train_data_loader = create_data_loader()

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_data_loader:
        optimizer.zero_grad()

        hidden = torch.zeros(1, inputs.size(0), hidden_size).to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

    # 在训练过程中计算并打印困惑度
    train_perplexity = compute_perplexity(model, train_data_loader, criterion)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Perplexity: {train_perplexity.item()}')
