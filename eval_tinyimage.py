from nbdt.model import SoftNBDT, HardNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torch.data.utils import Dataloader
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data.sampler import Sampler
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np
from explainer import Explainer
from RISE.evaluation import CausalMetric, auc, gkern
# from RISE.utils import *
import torch.nn as nn
import os

classes_cifar10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Given label number returns class name
def get_class_name(c):
    return classes_cifar10[c]

def get_class_name(c):
    # path.join(path.dirname(__file__), '..')
    labels = np.loadtxt('./tiny_synset_sorted.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

class Dummy():
    pass

# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)

preprocess = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

batch_size = 1
range_sample = range(200,300,10)

args = Dummy()

args.workers = 8
args.datadir = './tiny-imagenet-200/val/'
args.range = range(200, 300, 10)
args.input_size = (64, 64)
args.gpu_batch = 100

dataset = datasets.ImageFolder(args.datadir, preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))

#Load training data

hardModel = ResNet18(num_classes=200)
hardModel = HardNBDT(
  pretrained=True,
  dataset='TinyImagenet200',
  arch='ResNet18',
  model=hardModel)
hardModel = hardModel.cuda()

softModel = ResNet18(num_classes=200)
softModel = SoftNBDT(
  pretrained=True,
  dataset='TinyImagenet200',
  arch='ResNet18',
  model=softModel)
softModel = softModel.cuda()


explainer = Explainer(hardModel, (64, 64), 500)
explainer.generate_masks(1000, 8, 0.1, 'temp.npy')

klen = 11
ksig = 5
kern = gkern(klen, ksig)

blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

insertion = CausalMetric(softModel, 'ins', 64*8, substrate_fn=blur, n_classes=200, device=torch.device("cuda"))
deletion = CausalMetric(softModel, 'del', 64*8, substrate_fn=torch.zeros_like, n_classes=200, device=torch.device("cuda"))

mean_ins = 0
mean_del = 0

softmax = nn.Softmax(dim=1)
for i, data in enumerate(data_loader):
    if i >=50:
        break
    image, label = data
    logits, decisions, tree, names, path = hardModel.forward_with_decisions(image.cuda())
    probs = softmax(logits)
    cl = torch.argmax(probs, 1).cpu().numpy()[0]

    sal = explainer(image.cuda())

    print('Generating saliency maps for the decisions made: ')
    plt.figure(figsize=(10, 5))
    print(len(path[0]))
    if(len(path[0])>10):
        continue
    for j in range(len(path[0])):
        plt.subplot(int('42'+ str(j+1)))
        if names[0][path[0][j]] == '(generated)':
            plt.title('decision: {} with prob {:.3f}'.format(tree[0][path[0][j]], j+1), color='w')
        else:
            plt.title('decision: {}  '.format(j+1) + names[0][path[0][j]] + ' with prob {:.3f}'.format(tree[0][path[0][j]]), color='w')
        plt.imshow(sal[path[0][j]], alpha=0.5, cmap='jet')
        plt.axis('off')
    plt.savefig('./output_TinyIMGNET/TinyImageNet200_all_fig{}.jpg'.format(i), facecolor='black')
    plt.show()

    print('Generating final combined saliency map')
    final_sal = explainer.gen_final_sal(sal, path)
    img = image.clone()
    img = img.cpu().numpy()[0]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    for channel in range(3):
        img[channel] = img[channel]*std[channel] + mean[channel]
    
    img = img.transpose((1, 2, 0))
    img[img < 0] = 0
    
    img *=255
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.astype(np.uint8))
    plt.title(get_class_name(cl) + ' {:.3f}'.format(probs[0, cl]), color='w')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(final_sal, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig('./output_TinyIMGNET/figs/' + str(i) + 'TinyImageNet200_mean_sal.png', facecolor='black')


    if not os.path.exists('output_TinyIMGNET/insTIN{}'.format(i)):
        os.makedirs('output_TinyIMGNET/insTIN{}'.format(i))
    if not os.path.exists('output_TinyIMGNET/delTINTIN{}'.format(i)):
        os.makedirs('output_TinyIMGNET/delTIN{}'.format(i))

    scores2 = deletion.single_run(image, final_sal, verbose=1, save_to='./output_TinyIMGNET/delTIN{}/'.format(i))
    scores1 = insertion.single_run(image, final_sal, verbose=1, save_to='./output_TinyIMGNET/insTIN{}/'.format(i))

    
    

    mean_ins += auc(scores1)
    mean_del += auc(scores2)
    print('Insertion score so far: ', mean_ins/(i + 1))
    print('Deletion score so far: ', mean_del/(i +1))


fi = open('./output_TinyIMGNET/Results.txt', 'w+')
print('Insertion score: ', mean_ins/len(testloader), file=fi)
print('Deletion score: ', mean_del/len(testloader), file=fi)
fi.close()

    # print(final_sal)
    # break







