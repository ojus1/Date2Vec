from Model import Date2VecConvert
import torch

# Date2Vec embedder object
# Loads a pretrained model
d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

# Date-Time is 13:23:30 2019-7-23
x = torch.Tensor([13, 23, 30, 2019, 7, 23]).float()

# Get embeddings
embed = d2v(x)

print(embed, embed.shape)