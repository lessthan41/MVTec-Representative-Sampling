import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch import nn
from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


### t-SNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import KMeans


### config
K = 4           ### K-shot


### reset result
if os.path.exists(f"./result_k={K}.csv"):
    os.remove(f"./result_k={K}.csv")

### create images folder
if not os.path.exists(f"./images_k={K}"):
    os.makedirs(f"./images_k={K}")


class TrainData(Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.img_paths = self.img_dir.glob('*.png')
        self.img_paths = sorted(list(self.img_paths))
    
    def get_data_list(self):
        ret = []
        for i in range(len(self.img_paths)):
            ret.append(str(self.img_paths[i]))
        return ret

    def __len__(self):
        '''Return the number of sample
        '''
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        '''Map index `idx` to a sample, i.e., an image
        Args:
            idx: (int) index
        Return:
            img: (torch.FloatTensor) values in 0 ~ 1 and shaped [3, H, W]
        '''
        img_path = self.img_paths[idx]

        # load image
        img = Image.open(img_path)
        W, H = img.size
        img = img.convert('RGB')
        img = img.resize((512,512))
        img = tf.to_tensor(img)
        
        return img


class Net(nn.Module):
    def __init__(self):
        '''Defines parameters (what layers you gonna use)
        '''
        super().__init__() # necessary
        self.features = resnet18(pretrained=True) # changeable

    def forward(self, img_b):
        '''Define how layers are interact, that is, the forward function.
        In this network, img_b is passed to self.features

        Args:
            img_b: (torch.FloatTensor) input images (mini-batch), shaped [N, 3, H, W]
        Return:
            features: (torch.FloatTensor) output features of resnet-18
        '''
        features = self.features(img_b)
        # kpt_b = self.regression(features)
        return features


if __name__ == "__main__":
    for d in os.listdir("./data"):
        OBJ = d  ### object
        PATH = f"./data/{OBJ}/train/good"


        data = TrainData(PATH)
        img_list = data.get_data_list() 
        loader = DataLoader(data, batch_size=1)


        # cnt = 0
        fname = []
        features = []
        for img_b in iter(loader):
            device = "cuda"
            model = Net().to(device)
            img_b = img_b.to(device)
            model.eval()

            feat_b = model(img_b)
            features.append(feat_b.tolist()[0])
            
            # cnt += 1
            # if cnt > 30:
            #     break

        ### k-means
        X = np.array(features)
        kmeans = KMeans(n_clusters=K, random_state=0, max_iter=300, n_init="auto").fit(X)
        y = kmeans.labels_
        center = kmeans.cluster_centers_


        ### find nearest center image
        output = [-1 for x in range(len(center))]
        dist = [10e8 for x in range(len(center))]
        for i in range(len(X)):
            belongs = y[i]
            loss = np.linalg.norm(X[i] - center[belongs]) # L2 Loss
            if loss < dist[belongs]:
                output[belongs] = i
                dist[belongs] = loss


        ### get output files
        output_fname = [img_list[i] for i in output]
        with open(f"./result_k={K}.csv", "a") as fw:
            buffer = f"{OBJ}"
            for f in output_fname:
                buffer += f", {f}"
            fw.write(buffer + "\n")
        

        ### append center in X for viz
        X = np.append(X, center, axis=0)
        print(OBJ, ": ", X.shape)

        ### t-sne
        X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=5, verbose=1).fit_transform(X)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
        plt.clf()
        plt.scatter(X_norm[:-K, 0], X_norm[:-K, 1], c=plt.cm.tab20(y))
        plt.scatter(X_norm[-K:, 0], X_norm[-K:, 1], c="r", s=80)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"t-SNE for {OBJ}")
        plt.savefig(f"./images_k={K}/t-SNE_{OBJ}.png")
        plt.show()