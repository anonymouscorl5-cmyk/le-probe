import torch
import numpy as np
import argparse
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import umap


def interpolate_color(idx):
    """
    Implements the 'shade' coloring logic:
    4 anchor colors at specific intervals.
    0: C1, 7: C2, 14: C3, 21: C4, 31: C5 (optional end color)
    """
    # Anchor colors in RGB (Inferno style)
    anchors = [
        [255, 255, 190],  # Pale Yellow (Start)
        [255, 215, 0],  # Gold
        [255, 140, 0],  # Dark Orange
        [178, 34, 34],  # Firebrick Red
        [40, 0, 0],  # Deep Maroon (Goal)
    ]

    # Define the boundaries
    boundaries = [0, 7, 14, 21, 31]

    # Find which segment the index falls into
    segment = 0
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= idx <= boundaries[i + 1]:
            segment = i
            break

    # Interpolate within segment
    start_idx = boundaries[segment]
    end_idx = boundaries[segment + 1]

    t = (idx - start_idx) / (end_idx - start_idx) if end_idx != start_idx else 0

    c1 = np.array(anchors[segment])
    c2 = np.array(anchors[segment + 1])

    color = c1 * (1 - t) + c2 * t
    return f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"


def visualize_manifold(
    input_file, method="pca", output_html="manifold_3d.html", highlight_episodes=None
):
    print(f"🎨 Loading manifold data from {input_file}...")
    data = torch.load(input_file, weights_only=False)
    latents = data["latents"]
    indices = data["frame_indices"]
    ep_indices = data.get("episode_indices", None)

    # Ensure ep_indices exists
    if ep_indices is None:
        ep_indices = np.array([i // 32 for i in range(len(indices))])

    print(f"📉 Reducing dimensions using {method.upper()}...")
    if method.lower() == "pca":
        reducer = PCA(n_components=3)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=3, perplexity=30, max_iter=1000)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError(f"Unsupported method: {method}")

    reduced_data = reducer.fit_transform(latents)

    print(f"🖌 Applying color mapping...")
    colors = np.array([interpolate_color(idx) for idx in indices])

    # Create descriptive hover labels
    hover_text = np.array(
        [f"Ep: {ep_indices[i]} | Fr: {indices[i]}" for i in range(len(indices))]
    )

    # Split data for highlighting
    fig = go.Figure()

    if highlight_episodes:
        mask = np.isin(ep_indices, highlight_episodes)
        bg_mask = ~mask

        # Background Trace
        fig.add_trace(
            go.Scatter3d(
                x=reduced_data[bg_mask, 0],
                y=reduced_data[bg_mask, 1],
                z=reduced_data[bg_mask, 2],
                mode="markers",
                name="Manifold",
                marker=dict(size=2, color=colors[bg_mask], opacity=0.4),
                text=hover_text[bg_mask],
                hoverinfo="text",
            )
        )

        # Highlight Trace
        fig.add_trace(
            go.Scatter3d(
                x=reduced_data[mask, 0],
                y=reduced_data[mask, 1],
                z=reduced_data[mask, 2],
                mode="markers",
                name=f"Highlighted Ep: {highlight_episodes}",
                marker=dict(
                    size=3,
                    color=colors[mask],
                    opacity=1.0,
                    line=dict(color="black", width=3),
                ),
                text=hover_text[mask],
                hoverinfo="text",
            )
        )
    else:
        # Single trace if no highlights
        fig.add_trace(
            go.Scatter3d(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                z=reduced_data[:, 2],
                mode="markers",
                name="Manifold",
                marker=dict(size=3, color=colors, opacity=0.6),
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"LeWM Latent Manifold (3D {method.upper()})",
        scene=dict(xaxis_title="Comp 1", yaxis_title="Comp 2", zaxis_title="Comp 3"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    print(f"💾 Saving visualization to {output_html}...")
    fig.write_html(output_html)
    print(f"✨ Done! Open {output_html} to view the manifold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="manifold_data.pt")
    parser.add_argument(
        "--method", type=str, choices=["pca", "tsne", "umap"], default="pca"
    )
    parser.add_argument("--output", type=str, default="manifold_3d.html")
    parser.add_argument(
        "--highlight", type=int, nargs="+", help="Episode IDs to highlight"
    )
    args = parser.parse_args()

    visualize_manifold(args.input, args.method, args.output, args.highlight)
