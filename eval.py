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

import matplotlib.pyplot as plt
import numpy as np
from explainer import Explainer
from RISE.evaluation import CausalMetric, auc, gkern
import torch.nn as nn

classes_cifar10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Given label number returns class name
def get_class_name(c):
    return classes_cifar10[c]


# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)

transform = transforms.Compose([
  transforms.Resize(32),
  transforms.CenterCrop(32),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 1
range_sample = range(0,10)

#Load training data
trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                        download=False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
#Loaad testing data
testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                       download=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1, pin_memory=True, sampler=RangeSampler(range_sample))

model = wrn28_10_cifar10()
model = HardNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model)
model = model.cuda()

explainer = Explainer(model, (32, 32), 500)
explainer.generate_masks(1000, 8, 0.1, 'temp.npy')

klen = 11
ksig = 5
kern = gkern(klen, ksig)

blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

insertion = CausalMetric(model, 'ins', 32*8, substrate_fn=blur, n_classes=10, device=torch.device("cuda"))
deletion = CausalMetric(model, 'del', 32*8, substrate_fn=torch.zeros_like, n_classes=10, device=torch.device("cuda"))

mean_ins = 0
mean_del = 0

softmax = nn.Softmax(dim=1)
for i, data in enumerate(testloader):
    if i >=50:
        break
    image, label = data
    logits, decisions, tree, names, path = model.forward_with_decisions(image.cuda())
    probs = softmax(logits)
    cl = torch.argmax(probs, 1).cpu().numpy()[0]

    sal = explainer(image.cuda())

    print('Generating saliency maps for the decisions made: ')
    plt.figure(figsize=(10, 5))
    for i in range(len(path[0])):
        plt.subplot(int('22'+ str(i+1)))
        plt.title(names[0][path[0][i]] + ' {:.3f}'.format(tree[0][path[0][i]]))
        plt.imshow(sal[path[0][i]], alpha=0.5, cmap='jet')
        plt.axis('off')

    plt.show()

    print('Generating final combined saliency map')
    final_sal = explainer.gen_final_sal(sal, path)
    img = image.clone()
    img = img.cpu().numpy()[0]
    mean = [0.4914, 0.4822, 0.4465] 
    std = [0.2023, 0.1994, 0.2010]

    for channel in range(3):
        img[channel] = img[channel]*std[channel] + mean[channel]
    
    img = img.transpose((1, 2, 0))
    img[img < 0] = 0
    
    img *=255
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.astype(np.uint8))
    plt.title(classes_cifar10[cl] + ' {:.3f}'.format(probs[0, cl]))
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(final_sal, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig('/home/siddhant/figs/' + str(i) + 'sal.png')

    scores2 = deletion.single_run(image, final_sal, verbose=1, save_to='/home/siddhant/figs/del/')
    scores1 = insertion.single_run(image, final_sal, verbose=1, save_to='/home/siddhant/figs/ins/')

    mean_ins += auc(scores1)
    mean_del += auc(scores2)
    print('Insertion score so far: ', mean_ins/(i + 1))
    print('Deletion score so far: ', mean_del/(i +1))


print('Insertion score: ', mean_ins/len(testloader))
print('Deletion score: ', mean_del/len(testloader))

    # print(final_sal)
    break







