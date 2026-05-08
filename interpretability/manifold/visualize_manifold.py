import torch
import numpy as np
import argparse
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

def interpolate_color(idx):
    """
    Implements the 'shade' coloring logic:
    4 anchor colors at specific intervals.
    0: C1, 7: C2, 14: C3, 21: C4, 31: C5 (optional end color)
    """
    # Anchor colors in RGB
    anchors = [
        [75, 0, 130],   # Indigo (C1)
        [0, 128, 128],  # Teal (C2)
        [255, 215, 0],  # Gold (C3)
        [220, 20, 60],  # Crimson (C4)
        [128, 0, 128]   # Purple (C5 - Final phase)
    ]
    
    # Define the boundaries
    boundaries = [0, 7, 14, 21, 31]
    
    # Find which segment the index falls into
    segment = 0
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= idx <= boundaries[i+1]:
            segment = i
            break
            
    # Interpolate within segment
    start_idx = boundaries[segment]
    end_idx = boundaries[segment+1]
    
    t = (idx - start_idx) / (end_idx - start_idx) if end_idx != start_idx else 0
    
    c1 = np.array(anchors[segment])
    c2 = np.array(anchors[segment+1])
    
    color = c1 * (1 - t) + c2 * t
    return f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})'

def visualize_manifold(input_file, method="pca", output_html="manifold_3d.html"):
    print(f"🎨 Loading manifold data from {input_file}...")
    data = torch.load(input_file)
    latents = data["latents"]
    indices = data["frame_indices"]
    
    print(f"📉 Reducing dimensions using {method.upper()}...")
    if method.lower() == "pca":
        reducer = PCA(n_components=3)
    else:
        reducer = TSNE(n_components=3, perplexity=30, n_iter=1000)
        
    reduced_data = reducer.fit_transform(latents)
    
    print(f"🖌 Applying color mapping...")
    colors = [interpolate_color(idx) for idx in indices]
    
    # Create the 3D Scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        z=reduced_data[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.6
        ),
        text=[f"Frame: {idx}" for idx in indices],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=f"LeWM Latent Manifold (3D {method.upper()})",
        scene=dict(
            xaxis_title='Comp 1',
            yaxis_title='Comp 2',
            zaxis_title='Comp 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    print(f"💾 Saving visualization to {output_html}...")
    fig.write_html(output_html)
    print(f"✨ Done! Open {output_html} to view the manifold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="le-probe/interpretability/manifold/manifold_data.pt")
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="pca")
    parser.add_argument("--output", type=str, default="le-probe/interpretability/manifold/manifold_3d.html")
    args = parser.parse_args()
    
    visualize_manifold(args.input, args.method, args.output)
