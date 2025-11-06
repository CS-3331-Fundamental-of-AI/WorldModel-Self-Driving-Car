# HANOI-WORLD — Observation Module Prototype
### CS-3331 World-Model Autonomous Vehicle Project

This repository contains the **baseline multimodal Observation Module** for the HANOI-WORLD world-model architecture.  
It processes **image, text, trajectory, graph, and action inputs** and encodes them into unified latent representations for generative simulation and controller training.

The prototype is designed for:

✅ Testing each encoder individually  
✅ Visualizing multimodal latent structures  
✅ Preparing for JEPA-style predictive world models  
✅ Building a solid foundation for the full HANOI-WORLD system  

---

## 📁 Folder Structure

CS-3331-WorldModel-AV/
│
├── observation_module/ # Main code for encoders + observe.py tool
│ ├── image_encoder.py # CNN + CLIP-style projection
│ ├── text_encoder.py # Token encoder + Transformer
│ ├── action_encoder.py # MLP encoder
│ ├── graph_encoder.py # Lightweight GCN + Transformer
│ ├── trajectory_encoder.py # CNN + Transformer temporal encoder
│ ├── observe.py # Multimodal inference & visualization tool
│
├── data/ # Sample multimodal input data
│ ├── sample_img.jpg
│ ├── sample_graph.json
│ ├── sample_traj.csv
│
├── outputs/ # Auto-generated latent results + visualizations
│ ├── (empty by default)
│
├── run_demo.py # Minimal encoder test using dummy data
├── requirements.txt # Python dependencies
└── README.md


---

## 🛠️ Installation & Setup

### ✅ 1. Clone the repository
```bash
git clone https://github.com/DTJ-Tran/CS-3331-WorldModel-AV.git
cd CS-3331-WorldModel-AV

✅ 2. Create and activate a virtual environment
Windows
python -m venv venv
.\venv\Scripts\activate


Mac/Linux
python3 -m venv venv
source venv/bin/activate


✅ 3. Install dependencies
pip install -r requirements.txt

🚀 Running the Demo (run_demo.py)
This demo runs all encoders using dummy random data to verify that the module works.

✅ One-line command:
python observation_module/run_demo.py

✅ Expected Image Output
![Demo Output](demo.png)


✅ Expected Console Output
Image latent: torch.Size([1, 256])
Text latent: torch.Size([1, 256])
Actions latent: torch.Size([1, 128])
Graph latent: torch.Size([1, 256])
Trajectory latent: torch.Size([1, 256])
Demo completed successfully!
No visualization is produced in demo mode.

🔍 Running the Full Observation Tool (observe.py)
Processes real multimodal data and generates 6 visualization files + latent dump.

✅ One-line command:
python observation_module/observe.py --image data/sample_img.jpg --text "turn left at the intersection" --graph data/sample_graph.json --traj data/sample_traj.csv --actions "0.1,0.3,0.0" --save_vis --output observe_output.pkl

✅ Expected Outputs (generated in /outputs/)
outputs/
│
├── image_vis.png
├── graph_embedding_vis.png
├── trajectory_vis.png
├── latent_similarity_heatmap.png
├── latent_distribution.png
└── multimodal_dashboard.png

Also saved:
observe_output.pkl — all latent vectors

✅ Example Console Output
Image processed: torch.Size([1, 256])
Text processed: torch.Size([1, 256])
Actions processed: torch.Size([1, 128])
Graph processed: torch.Size([1, 256])
Trajectory processed: torch.Size([1, 256])

✅ Latents saved to: outputs/observe_output.pkl
✅ Saved: outputs/image_vis.png
✅ Saved: outputs/graph_embedding_vis.png
✅ Saved: outputs/trajectory_vis.png
✅ Saved: outputs/latent_similarity_heatmap.png
✅ Saved: outputs/latent_distribution.png
✅ Saved: outputs/multimodal_dashboard.png


✅ Using Your Own Data
Replace paths as needed:
python observation_module/observe.py --image my_img.jpg --text "go straight" --graph hanoi_map.json --traj motorbike_traj.csv --actions "0.0,1.0,0.0" --save_vis --output my_latents.pkl

Input Format Requirements
✅ Image
.jpg or .png
Any resolution

✅ Graph JSON
{
  "nodes": [[x,y,z], ...],
  "adj": [[0,1,0...], ...]
}

✅ Trajectory CSV
x,y

✅ Actions
Comma-separated floats:
"steering, throttle, brake"

✅ .gitignore (recommended)
venv/
__pycache__/
**/__pycache__/
outputs/
*.pkl
*.png
*.pt
*.pth
.DS_Store