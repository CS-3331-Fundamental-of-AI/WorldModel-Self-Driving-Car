import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from action_encoder import ActionEncoder
from graph_encoder import GraphEncoder
from trajectory_encoder import TrajectoryEncoder

# Dummy inputs
img = torch.randn(1, 3, 128, 128)
tokens = torch.randint(0, 2000, (1, 12))
actions = torch.randn(1, 3)
graph = torch.randn(1, 10, 32)
adj = torch.eye(10).unsqueeze(0)
traj = torch.randn(1, 20, 3)

# Create encoders
img_enc = ImageEncoder()
txt_enc = TextEncoder()
act_enc = ActionEncoder()
graph_enc = GraphEncoder()
traj_enc = TrajectoryEncoder()

# Forward
img_lat, img_clip = img_enc(img)
txt_lat = txt_enc(tokens)
act_lat = act_enc(actions)
graph_lat, graph_nodes = graph_enc(graph, adj)
traj_lat = traj_enc(traj)

print("Image latent:", img_lat.shape)
print("Text latent:", txt_lat.shape)
print("Action latent:", act_lat.shape)
print("Graph latent:", graph_lat.shape)
print("Trajectory latent:", traj_lat.shape)

# Visualize graph nodes
nodes = graph_nodes[0].detach().numpy()
node_2d = PCA(n_components=2).fit_transform(nodes)
plt.scatter(node_2d[:,0], node_2d[:,1])
plt.title("Graph Node Embeddings (PCA)")
plt.show()

print("Demo completed.")
