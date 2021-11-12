import models.ms as ms
import models.mv as mv

# 训练参数
use_gpu = True  # 是否使用GPU
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
std = 0.001  # 输出涂抹标准差
batch_size = 32  # batch块大小
initial_epochs = 100  # 训初始练轮数
update_epochs = 60  # 模型每次更新训练轮数
learning_rate = 0.01  # 学习率
gamma = 10
T = 30  # 模型更新轮数
U = 59900  # 未标记数据大小
sigma_0 = 0.999  # σ_0
sigma_os = 0.01  # σ_os
initial_size = 100  # 初始数据集大小
# 路径参数
data_root = './custom_dataset'
mnist_path = data_root + '/MNIST'
save_dir = './experiment'
tri_net_save_path = save_dir + "/tri-net"
# 数据集
dataset = "MNIST"
feature_extractor_dict = {'MNIST': ms.MS()}
label_predictor_dict = {'MNIST': [mv.M1(), mv.M2(), mv.M3()]}
domain_predictor_dict = {'MNIST': ""}
