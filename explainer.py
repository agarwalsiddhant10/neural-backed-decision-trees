from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torch.data.utils import Dataloader
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from RISE.explanations import RISE

class Explainer(RISE):
    def forward(self, x):
        with torch.no_grad():
            N = self.N
            _, _, H, W = x.size()
            stack = torch.mul(self.masks, x.data)
            
            P = []
            for i in range(0, N, self.gpu_batch):
                inp_batch = stack[i:min(i + self.gpu_batch, N)]
                outputs = self.model.rules.forward_nodes(inp_batch)
                for o in ouputs:
                    P.append(o['probs'][0])
                    P.append(o['probs'][1])

            sal = torch.matmul(P.transpose(0, 1), self.masks.view(N, H*W))
            del p
            del stack
            sal.view((len(P), H, W))
            sal = sal / N / self.p1

            return sal

    def gen_final_sal(sal, path):
        final_sal = np.zeros(sal.shape[1], sal.shape[2])
        weight = 1
        for i in path:
            final_sal += weight*sal[i]
            weight += 0.2
        

        return final_sal/weight

    


