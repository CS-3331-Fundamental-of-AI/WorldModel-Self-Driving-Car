"""
observe.py — Multimodal Observation Module Demo

Run example (Windows one-line):
python observation_module/observe.py --image data/sample_img.jpg --text "turn left at the intersection" --graph data/sample_graph.json --traj data/sample_traj.csv --actions "0.1,0.3,0.0" --save_vis
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from action_encoder import ActionEncoder
from graph_encoder import GraphEncoder
from trajectory_encoder import TrajectoryEncoder



# -------------------------------------------------------------------------
# DATA LOADING HELPERS
# -------------------------------------------------------------------------

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    return tensor.unsqueeze(0), np.array(img)


def load_text(text):
    words = text.lower().split()
    vocab = {w: i+1 for i, w in enumerate(set(words))}
    token_ids = [vocab[w] for w in words]
    return torch.tensor([token_ids])


def load_trajectory(path):
    data = np.loadtxt(path, delimiter=',')
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0), data


def load_graph(path):
    with open(path) as f:
        g = json.load(f)

    nodes = torch.tensor(g["nodes"], dtype=torch.float32)
    adj = torch.tensor(g["adj"], dtype=torch.float32)
    return nodes.unsqueeze(0), adj.unsqueeze(0), g["nodes"], g["adj"]


def load_actions(s):
    arr = [float(x) for x in s.split(',')]
    return torch.tensor([arr], dtype=torch.float32)


# -------------------------------------------------------------------------
# VISUALIZATION PACK (including multimodal dashboard)
# -------------------------------------------------------------------------

def visualize_results(outputs, OUTPUT_DIR):

    # ---------------------------------------------------------------------
    # IMAGE PREVIEW
    # ---------------------------------------------------------------------
    if "raw_image" in outputs:
        plt.figure(figsize=(4, 4))
        plt.imshow(outputs["raw_image"])
        plt.axis("off")
        out = os.path.join(OUTPUT_DIR, "image_vis.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", out)

    # ---------------------------------------------------------------------
    # GRAPH VISUALIZATION
    # ---------------------------------------------------------------------
    if "graph_nodes" in outputs:

        nodes = outputs["graph_nodes"][0]   # [N, D]
        adj = outputs["adj_matrix"][0]      # [N, N]
        N = nodes.shape[0]

        node_2d = PCA(n_components=2).fit_transform(nodes)

        plt.figure(figsize=(6, 6))
        # draw edges
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    plt.plot([node_2d[i, 0], node_2d[j, 0]],
                             [node_2d[i, 1], node_2d[j, 1]],
                             "k-", alpha=0.3)

        # draw nodes
        plt.scatter(node_2d[:, 0], node_2d[:, 1], c="red")
        for i in range(N):
            plt.text(node_2d[i, 0], node_2d[i, 1], f"N{i}")

        plt.title("Graph Embedding (PCA)")
        out = os.path.join(OUTPUT_DIR, "graph_embedding_vis.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", out)

    # ---------------------------------------------------------------------
    # TRAJECTORY VISUALIZATION
    # ---------------------------------------------------------------------
    if "raw_traj" in outputs:

        traj = outputs["raw_traj"]

        plt.figure(figsize=(5, 5))
        plt.plot(traj[:, 0], traj[:, 1], marker="o")
        plt.title("Trajectory Path")
        plt.xlabel("X")
        plt.ylabel("Y")

        out = os.path.join(OUTPUT_DIR, "trajectory_vis.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", out)

    # ---------------------------------------------------------------------
    # ----- LATENT SIMILARITY MATRIX -----
    # ---------------------------------------------------------------------
    
    # Project all latents to a shared vector length
    TARGET_DIM = 256

    def normalize_dim(vec, target_dim=TARGET_DIM):
        v = torch.tensor(vec).flatten()
        if v.numel() < target_dim:
            # pad zeros
            pad = torch.zeros(target_dim - v.numel())
            v = torch.cat([v, pad])
        elif v.numel() > target_dim:
            # truncate
            v = v[:target_dim]
        return v

    def cos(a, b):
        a = normalize_dim(a)
        b = normalize_dim(b)
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return float(torch.dot(a, b).item())

    modalities = ["image_latent", "text_latent", "action_latent",
                  "graph_latent", "trajectory_latent"]

    avail = [m for m in modalities if m in outputs]
    size = len(avail)
    sim = np.zeros((size, size))

    for i, m1 in enumerate(avail):
        for j, m2 in enumerate(avail):
            sim[i, j] = cos(outputs[m1], outputs[m2])

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(size), avail, rotation=45)
    plt.yticks(range(size), avail)
    plt.title("Latent Similarity Heatmap")

    out = os.path.join(OUTPUT_DIR, "latent_similarity_heatmap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out)


    # ---------------------------------------------------------------------
    # LATENT DISTRIBUTION
    # ---------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for m in avail:
        plt.plot(outputs[m].flatten(), label=m)
    plt.title("Latent Magnitude Distribution")
    plt.legend()

    out = os.path.join(OUTPUT_DIR, "latent_distribution.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out)

    # ---------------------------------------------------------------------
    # MULTIMODAL DASHBOARD (COMBINES EVERYTHING)
    # ---------------------------------------------------------------------

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # IMAGE
    if "raw_image" in outputs:
        axs[0, 0].imshow(outputs["raw_image"])
        axs[0, 0].set_title("Input Image")
        axs[0, 0].axis("off")
    else:
        axs[0, 0].text(0.5, 0.5, "No image", ha="center")

    # GRAPH
    if "graph_nodes" in outputs:
        node_2d = PCA(n_components=2).fit_transform(outputs["graph_nodes"][0])
        axs[0, 1].scatter(node_2d[:, 0], node_2d[:, 1], c="red")
        for i in range(node_2d.shape[0]):
            axs[0, 1].text(node_2d[i, 0], node_2d[i, 1], f"N{i}")
        axs[0, 1].set_title("Graph PCA View")
    else:
        axs[0, 1].text(0.5, 0.5, "No graph", ha="center")

    # TRAJECTORY
    if "raw_traj" in outputs:
        traj = outputs["raw_traj"]
        axs[1, 0].plot(traj[:, 0], traj[:, 1], marker="o")
        axs[1, 0].set_title("Trajectory Path")
    else:
        axs[1, 0].text(0.5, 0.5, "No trajectory", ha="center")

    # LATENT SIMILARITY
    axs[1, 1].imshow(sim, cmap="viridis")
    axs[1, 1].set_xticks(range(size))
    axs[1, 1].set_xticklabels(avail, rotation=45)
    axs[1, 1].set_yticks(range(size))
    axs[1, 1].set_yticklabels(avail)
    axs[1, 1].set_title("Latent Similarity")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "multimodal_dashboard.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out)



# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main(args):

    # fixed absolute save location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n🚀 Running Observation Module...\n")
    print("Working directory:", os.getcwd())
    print("Output directory:", OUTPUT_DIR, "\n")

    img_enc = ImageEncoder()
    txt_enc = TextEncoder()
    act_enc = ActionEncoder()
    graph_enc = GraphEncoder()
    traj_enc = TrajectoryEncoder()

    outputs = {}

    # IMAGE
    if args.image:
        img_tensor, raw_img = load_image(args.image)
        img_lat, img_clip = img_enc(img_tensor)
        outputs["image_latent"] = img_lat.detach().numpy()
        outputs["image_clip"] = img_clip.detach().numpy()
        outputs["raw_image"] = raw_img
        print("Image processed:", img_lat.shape)

    # TEXT
    if args.text:
        tokens = load_text(args.text)
        txt_lat = txt_enc(tokens)
        outputs["text_latent"] = txt_lat.detach().numpy()
        print("Text processed:", txt_lat.shape)

    # ACTIONS
    if args.actions:
        act = load_actions(args.actions)
        act_lat = act_enc(act)
        outputs["action_latent"] = act_lat.detach().numpy()
        print("Actions processed:", act_lat.shape)

    # GRAPH
    if args.graph:
        graph_nodes, adj, raw_nodes, raw_adj = load_graph(args.graph)
        graph_lat, node_latents = graph_enc(graph_nodes, adj)
        outputs["graph_latent"] = graph_lat.detach().numpy()
        outputs["graph_nodes"] = node_latents.detach().numpy()
        outputs["adj_matrix"] = adj.detach().numpy()
        print("Graph processed:", graph_lat.shape)

    # TRAJECTORY
    if args.traj:
        traj_tensor, raw_traj = load_trajectory(args.traj)
        traj_lat = traj_enc(traj_tensor)
        outputs["trajectory_latent"] = traj_lat.detach().numpy()
        outputs["raw_traj"] = raw_traj
        print("Trajectory processed:", traj_lat.shape)

    # Save latents (.pkl)
    out_file = os.path.join(OUTPUT_DIR, args.output)
    import pickle
    with open(out_file, "wb") as f:
        pickle.dump(outputs, f)
    print("\n✅ Latents saved to:", out_file)

    # VISUALIZATION PACK
    if args.save_vis:
        visualize_results(outputs, OUTPUT_DIR)

    print("\n🎉 Observation module completed successfully.\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Observation Module on real data.")

    parser.add_argument("--image", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--graph", type=str)
    parser.add_argument("--traj", type=str)
    parser.add_argument("--actions", type=str)
    parser.add_argument("--output", type=str, default="observe_output.pkl")
    parser.add_argument("--save_vis", action="store_true")

    args = parser.parse_args()
    main(args)
