import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

from __gpu import gpu

class SiameseNetwork(nn.Module):

    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()
        
        #Pretrained model
        self.FeatureExtractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in self.FeatureExtractor.parameters():
            param.requires_grad = False

        #Get total number of output features
        out_features = self.FeatureExtractor.fc.out_features

        # Create an MLP (multi-layer perceptron) as the classification head.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.FeatureExtractor(img1)
        feat2 = self.FeatureExtractor(img2)
        
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
        combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(combined_features)
        return output
        
        

    

if __name__ == '__main__':
    device = gpu()
    siamese_network = SiameseNetwork()
    # Assuming input shape is (batch_size, channels, height, width)
    input1 = torch.randn(2048, 3, 224, 224) 
    input2 = torch.randn(2048, 3, 224, 224)
    model_summary = summary(siamese_network, input_data=[input1, input2], device=device)
    print(model_summary)

