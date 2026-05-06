# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import torch
import numpy as np
import time
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    import torchcodec

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


class LEWMDataPlugin(torch.utils.data.Dataset):
    """
    High-performance Direct Bypass plugin for LeRobot datasets.
    Bypasses the slow LeRobotDataset wrapper and speaks directly to Parquet and MP4 files.
    Optimized for 500+ FPS research throughput.
    """

    def __init__(
        self,
        repo_id,
        keys_to_load,
        num_steps=1,
        transform=None,
        use_virtual_actions=True,
    ):
        self.repo_id = repo_id
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform
        self.use_virtual_actions = use_virtual_actions

        # 1. Base Dataset Discovery
        self.lerobot_dataset = LeRobotDataset(repo_id)
        self.root = Path(self.lerobot_dataset.root)
        self.hf_dataset = self.lerobot_dataset.hf_dataset

        # 2. Key Mapping & Dim Detection
        self.key_map = {
            "pixels": "observation.images.world_center",
            "state": "observation.state",
            "proprio": "observation.state",
            "action": "action",
        }

        # 3. HIGH SPEED METADATA CACHE (Zero Parquet latency)
        print(f"🚀 Initializing Direct Bypass for {repo_id}...")
        self.episode_indices = torch.from_numpy(
            np.array(self.hf_dataset["episode_index"])
        )
        self.frame_indices = torch.from_numpy(np.array(self.hf_dataset["frame_index"]))

        self.cached_states = None
        if "observation.state" in self.hf_dataset.column_names:
            self.cached_states = torch.from_numpy(
                np.array(self.hf_dataset["observation.state"])
            )

        self.cached_actions = None
        self.has_native_actions = "action" in self.hf_dataset.column_names
        if self.has_native_actions:
            self.cached_actions = torch.from_numpy(np.array(self.hf_dataset["action"]))
            print("⚡ Using native action column from RAM cache.")

        # Progress / Rewards
        self.cached_progress = None
        self.has_progress = False

        reward_cols = ["progress_sparse", "progress", "reward"]
        for col in reward_cols:
            if col in self.hf_dataset.column_names:
                self.cached_progress = torch.from_numpy(np.array(self.hf_dataset[col]))
                self.has_progress = True
                print(f"📈 Using {col} column from RAM cache.")
                break

        # Fallback to side-car parquet file (common for research datasets on Colab)
        if not self.has_progress:
            reward_file = self.root / "progress_sparse.parquet"
            if not reward_file.exists():
                try:
                    print(f"📥 Downloading side-car reward file from {self.repo_id}...")
                    reward_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename="progress_sparse.parquet",
                        repo_type="dataset",
                    )
                    reward_file = Path(reward_path)
                except Exception as e:
                    print(f"⚠️ Could not download progress_sparse.parquet: {e}")

            if reward_file.exists():
                df = pd.read_parquet(reward_file)
                for col in reward_cols:
                    if col in df.columns:
                        self.cached_progress = torch.from_numpy(df[col].values).float()
                        self.has_progress = True
                        print(f"📈 Loaded {col} from side-car file: {reward_file.name}")
                        break

        # 4. LRU Decoder Cache (Worker-local)
        self._decoders = {}

    def _get_decoder(self, video_path):
        """Returns a cached VideoDecoder instance for the given path."""
        if video_path not in self._decoders:
            if not HAS_TORCHCODEC:
                raise RuntimeError(
                    "torchcodec is required for High-Performance Direct Bypass."
                )
            self._decoders[video_path] = torchcodec.decoders.VideoDecoder(
                str(video_path)
            )
        return self._decoders[video_path]

    def _get_video_path(self, episode_idx, image_key):
        """Constructs the direct file path for a specific episode and camera."""
        # Pattern verified: videos/{key}/chunk-000/file-{ep:03d}.mp4
        return (
            self.root
            / "videos"
            / image_key
            / "chunk-000"
            / f"file-{episode_idx:03d}.mp4"
        )

    def clear_cache(self):
        """Clears cached decoders to release file handles before multi-process forks."""
        self._decoders = {}

    def __len__(self):
        buffer = 1 if (self.use_virtual_actions and not self.has_native_actions) else 0
        return len(self.hf_dataset) - (self.num_steps + buffer)

    def __getitem__(self, idx):
        # 1. Episode Boundary Logic
        buffer = 1 if (self.use_virtual_actions and not self.has_native_actions) else 0
        ep_start = self.episode_indices[idx]
        ep_end = self.episode_indices[idx + self.num_steps + buffer - 1]

        if ep_start != ep_end:
            # Shift back to stay within ep_start
            idx = idx - self.num_steps
            if idx < 0:
                idx = 0

        # 2. Fetch Metadata (Instant RAM access)
        batch = {}
        fetch_len = self.num_steps
        if self.use_virtual_actions and not self.has_native_actions:
            fetch_len += 1

        # State/Proprio
        if self.cached_states is not None:
            state_seq = self.cached_states[idx : idx + fetch_len]
            batch["observation.state"] = state_seq[: self.num_steps]
            if self.use_virtual_actions and not self.has_native_actions:
                # Filter NaNs if any exist in the state signal
                diff = state_seq[1:] - state_seq[:-1]
                batch["action"] = torch.where(
                    torch.isnan(diff), torch.zeros_like(diff), diff
                )

        # Native Actions
        if self.has_native_actions:
            batch["action"] = self.cached_actions[idx : idx + self.num_steps]

        if self.has_progress:
            batch["progress"] = self.cached_progress[idx : idx + self.num_steps]

        # 3. Direct Video Decoding (High Performance)
        for target_key in self.keys_to_load:
            if "images" in target_key or target_key == "pixels":
                image_key = self.key_map.get(target_key, target_key)
                episode_idx = int(self.episode_indices[idx])
                frame_idx = int(self.frame_indices[idx])

                video_path = self._get_video_path(episode_idx, image_key)
                decoder = self._get_decoder(video_path)

                # Fetch the entire sequence in ONE call
                seq_indices = list(range(frame_idx, frame_idx + self.num_steps))
                # Torchcodec returns a FrameBatch object. Use .data to get the (T, C, H, W) tensor.
                frames = decoder.get_frames_at(indices=seq_indices)
                # Torchcodec returns [T, C, H, W] in RGB uint8 [0, 255] by default.
                # We simply use .data and ensure it's in byte format.
                batch[target_key] = frames.data.byte()

        # 4. Standard Plugin Post-Processing (Nesting/Transforms)
        nested_batch = self.nest_dict(batch)
        if self.transform:
            nested_batch = self.transform(nested_batch)

        final_batch = self.flatten_dict(nested_batch)

        # Add Aliases for Model Compatibility
        aliases = {}
        for k, v in final_batch.items():
            if "world_center" in k:
                aliases["pixels"] = v
            if "observation.state" in k:
                aliases["state"] = v
                aliases["proprio"] = v
        final_batch.update(aliases)

        return final_batch

    @staticmethod
    def nest_dict(flat_dict):
        nested_dict = {}
        for k, v in flat_dict.items():
            parts = k.split(".")
            d = nested_dict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = v
        return nested_dict

    @staticmethod
    def flatten_dict(nested_dict, parent_key="", sep="."):
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(LEWMDataPlugin.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_col_data(self, col_name):
        """Fast path for normalizer using RAM cache."""
        if col_name == "action" and self.cached_actions is not None:
            return self.cached_actions.numpy()
        if (
            col_name == "observation.state" or col_name == "state"
        ) and self.cached_states is not None:
            return self.cached_states.numpy()

        # Fallback for other columns
        return np.array(self.hf_dataset[col_name])

    def get_dim(self, col_name):
        """Fast path for dimension check."""
        source_key = self.key_map.get(col_name, col_name)
        if source_key == "action" and self.cached_actions is not None:
            return self.cached_actions.shape[-1]
        if source_key == "observation.state" and self.cached_states is not None:
            return self.cached_states.shape[-1]

        return len(self.hf_dataset[0][source_key])
