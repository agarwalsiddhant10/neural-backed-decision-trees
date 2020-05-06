from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torch.data.utils import Dataloader
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from RISE.explanations import RISE
import torch
import numpy as np

class Explainer(RISE):
    def forward(self, x):
        with torch.no_grad():
            N = self.N
            _, _, H, W = x.size()
            stack = torch.mul(self.masks, x.data)
            
            
            P = None
            for i in range(0, N, self.gpu_batch):
                inp_batch = stack[i:min(i + self.gpu_batch, N)]
                # print(inp_batch.size())
                _, _, tree, _, _ = self.model.forward_with_decisions(inp_batch)
                if P is None:
                    P = np.array(tree)
                else:
                    P = np.vstack((P, np.array(tree)))

            P = torch.from_numpy(P)
            sal = torch.matmul(P.transpose(0, 1).float().cuda(), self.masks.view(N, H*W))
            n = P.size(1)
            del P
            del stack
            sal = sal.view(n, H, W)
            sal = sal / N / self.p1

            return sal.cpu().numpy()

    def gen_final_sal(self,sal, path):
        final_sal = np.zeros((sal.shape[1], sal.shape[2]))
        weight = 1
        for i in path[0]:
            # print(i)
            # print(sal[i].shape)
            final_sal += weight*sal[i]
            weight += 0
        

        return final_sal/len(path[0])

    


