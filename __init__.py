from resNet import resnet12
from resNet import seresnet12

model_pool = [
    'convnet4',
    'resnet12',
    'seresnet12',
    'wrn_28_10',
]

model_dict = {
    'resnet12': resnet12,
    'seresnet12': seresnet12,
}