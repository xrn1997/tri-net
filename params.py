import models.ms as ms
import models.mv as mv

# 训练参数
use_gpu = True  # 是否使用GPU
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
std = 0.001  # 输出涂抹标准差
batch_size = 16  # batch块大小
epochs = 20  # 训练轮数
learning_rate = 0.01  # 学习率
gamma = 10
# 路径参数
data_root = './data'
mnist_path = data_root + '/MNIST'
save_dir = './experiment'
tri_net_save_path = save_dir + "/tri-net"
# 数据集
dataset = "MNIST"
feature_extractor_dict = {'MNIST': ms.MS()}
label_predictor_dict = {'MNIST': [mv.M1(), mv.M2(), mv.M3()]}
domain_predictor_dict = {'MNIST': ""}
