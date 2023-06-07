# 这段改成我的配置文件路径
'''
_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]
'''

_base_ = [
    '../../../config_for_fruits.py',
]


# model settings-------------------
# 由于我们的分类只有30类，所以num_classes=30
model = dict(head=dict(num_classes=30))

# schedule settings-----------------
optim_wrapper = dict(optimizer=dict(weight_decay=0.0005))


# data settings---------------------

# 数据集类型为CustomDataset
dataset_type = 'CustomDataset'

# 同理，我们只有30类
data_preprocessor = dict(num_classes=30)

# 训练集路径改成自己的
train_dataloader = dict(dataset=dict(data_root='data/fruit30_train'))

# 验证集也是的
val_dataloader = dict(dataset=dict(data_root='data/fruit30_train'))

# 再配置文件中添加一个test_dataloader，因为原来的配置文件中没有

# 规划配置
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
)


# 由于是微调，只需少量训练即可，我设置为7
train_cfg = dict(by_epoch=True, max_epochs=7, val_interval=1)

# lr调小一点
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))


# 加载预训练权重
init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth')
