import streamlit as st
import numpy as np
import json
import os
import time

# --- Path Stabilization ---
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gr1_config import COMPACT_WIRE_JOINTS
from inference_http import InferenceHTTPClient, TELEOP_PATH, TELEOP_TIMEOUT_S

st.set_page_config(page_title="Le-Probe: Teleop Dashboard", layout="wide")


# --- Teleop HTTP client (MuJoCo server on :5556) ---
DEFAULT_TELEOP_URL = "http://127.0.0.1:5556"


@st.cache_resource
def get_teleop_client(base_url: str = DEFAULT_TELEOP_URL):
    return InferenceHTTPClient(
        base_url,
        timeout_s=TELEOP_TIMEOUT_S,
        endpoint=TELEOP_PATH,
    )


teleop_url = st.sidebar.text_input(
    "Teleop server URL",
    value=DEFAULT_TELEOP_URL,
    help="Local default; use ngrok URL if sim runs remotely",
)
client = get_teleop_client(teleop_url)

# --- Initialize Session State ---
if "target_buffer" not in st.session_state:
    st.session_state.target_buffer = np.full(32, np.nan, dtype=np.float32)

if "staging_buffer" not in st.session_state:
    st.session_state.staging_buffer = np.full(32, np.nan, dtype=np.float32)

if "total_episodes" not in st.session_state:
    st.session_state.total_episodes = 0
if "upload_queue" not in st.session_state:
    st.session_state.upload_queue = 0
if "batch_status" not in st.session_state:
    st.session_state.batch_status = 0
if "ik_phase" not in st.session_state:
    st.session_state.ik_phase = None
if "physics" not in st.session_state:
    st.session_state.physics = {"cube_z": 0.0, "is_grasping": False, "target_dist": 0.0}
if "lewm_reward" not in st.session_state:
    st.session_state.lewm_reward = None

# --- Load Default Active Joints ---
if "active_joints" not in st.session_state:
    st.session_state.active_joints = set()
    # Load default active joints from IK whitelist
    base_path = ROOT_DIR  # Point to root for assets
    with open(f"{base_path}/ik_joints.txt", "r") as f:
        default_joint_names = [
            line.strip().split("#")[0].strip() for line in f if line.strip()
        ]
        for name in default_joint_names:
            if name in COMPACT_WIRE_JOINTS:
                idx = COMPACT_WIRE_JOINTS.index(name)
                st.session_state.active_joints.add(idx)
                st.session_state[f"input_{idx}"] = 0.0
                st.session_state.staging_buffer[idx] = 0.0


def send_command(payload, *, update_lewm_reward: bool = True):
    try:
        data = client.command(payload)
        st.session_state.upload_queue = data.get("upload_queue", 0)
        st.session_state.total_episodes = data.get("total_episodes", 0)
        st.session_state.batch_status = data.get("batch_status", 0)
        if "physics" in data:
            st.session_state.physics = data["physics"]
        if update_lewm_reward:
            if "lewm_reward" in data:
                st.session_state.lewm_reward = data["lewm_reward"]
            elif "lewm_reward_error" in data:
                st.session_state.lewm_reward = {"error": data["lewm_reward_error"]}
        return data
    except Exception as e:
        st.error(f"Teleop HTTP error: {e}")
        return None


# Automated status refresh on EVERY rerun to ensure parity with server
try:
    data = send_command({"command": "poll_status"}, update_lewm_reward=False)
    if data:
        st.session_state.total_episodes = data.get(
            "total_episodes", st.session_state.total_episodes
        )
        st.session_state.upload_queue = data.get(
            "upload_queue", st.session_state.upload_queue
        )
        st.session_state.batch_status = data.get(
            "batch_status", st.session_state.batch_status
        )
except Exception as e:
    print("Error while setting initial total episodes", e)


def sync_ui_to_joints(joints):
    """Updates the UI session state to match a 32-DOF joint vector."""
    st.session_state.staging_buffer = np.array(joints, dtype=np.float32)
    st.session_state.target_buffer = np.copy(st.session_state.staging_buffer)

    # Automatically activate sliders for important joints
    for idx, val in enumerate(joints):
        clipped_val = float(np.clip(val, -1.0, 1.0))
        if abs(clipped_val) > 1e-4:
            if idx not in st.session_state.active_joints:
                st.session_state.active_joints.add(idx)
            # Update the key associated with the slider
            st.session_state[f"input_{idx}"] = clipped_val


def handle_reset():
    resp = send_command({"command": "reset"})
    if resp and "joints" in resp:
        sync_ui_to_joints(resp["joints"])
        st.session_state.last_msg = ("Randomized! Sliders synced.", "🎲")


def handle_wild_reset():
    resp = send_command({"command": "wild_randomize"})
    if resp and "joints" in resp:
        sync_ui_to_joints(resp["joints"])
        st.session_state.last_msg = ("WILD RANDOMIZED! Manifold expanded.", "🌀")


def handle_submit():
    """Send target after sliders; must use on_click so reward metrics refresh same run."""
    final_packet = [float("nan")] * 32
    for idx in st.session_state.active_joints:
        final_packet[idx] = float(st.session_state.staging_buffer[idx])
    send_command({"target": final_packet})
    st.session_state.last_msg = ("Sent Action!", "🚀")


def handle_ik_pickup(offset_cm):
    st.session_state.ik_phase = 0
    st.session_state.ik_offset = offset_cm


def handle_start_recording(task_name):
    resp = send_command({"command": "start_recording", "task": task_name})
    if resp and resp.get("status") == "recording_started":
        st.session_state.is_recording = True
        st.session_state.last_msg = ("Recording Started!", "🔴")


def send_stop_recording():
    return send_command({"command": "stop_recording"})


def send_discard_recording():
    return send_command({"command": "discard_recording"})


def apply_snapshot(snapshot_dict):
    """Maps name-based snapshot to 32-DOF and syncs UI/Server."""
    normalized = snapshot_dict.get("normalized", {})
    target = np.zeros(32, dtype=np.float32)

    for name, val in normalized.items():
        if name in COMPACT_WIRE_JOINTS:
            idx = COMPACT_WIRE_JOINTS.index(name)
            # SAFETY CLIP: Ensure value fits in [-1, 1] slider range
            clipped_val = float(np.clip(val, -1.0, 1.0))
            target[idx] = clipped_val
            # Update the EXACT key used by the slider to force a UI refresh
            st.session_state[f"input_{idx}"] = clipped_val

    # 1. Send Cube Pose FIRST to stage the scene
    cube_qpos = snapshot_dict.get("cube_qpos", [])
    if cube_qpos:
        send_command({"command": "set_cube_pose", "pose": cube_qpos})

    # 2. Update buffers and send robot target
    st.session_state.staging_buffer = target.copy()
    st.session_state.target_buffer = target.copy()
    send_command({"target": target.tolist()})


def home_all():
    st.session_state.target_buffer.fill(0.0)
    st.session_state.staging_buffer.fill(0.0)
    for idx in st.session_state.active_joints:
        st.session_state[f"input_{idx}"] = 0.0


def clear_all():
    for idx in st.session_state.active_joints:
        if f"input_{idx}" in st.session_state:
            del st.session_state[f"input_{idx}"]
    st.session_state.active_joints.clear()
    st.session_state.target_buffer.fill(np.nan)
    st.session_state.staging_buffer.fill(np.nan)


# --- Phase Machine Sync Consumer ---
# This ensures that IK phase updates are applied to widgets BEFORE they are instantiated,
# while the actual phase command happened at the end of the previous run (after instantiation).
if "pending_sync" in st.session_state:
    sync_ui_to_joints(st.session_state.pending_sync)
    del st.session_state.pending_sync

st.title("Le-Probe: Teleop Dashboard")

# Show any pending toasts from callbacks
if "last_msg" in st.session_state:
    msg, icon = st.session_state.last_msg
    st.toast(msg, icon=icon)
    del st.session_state.last_msg

# --- Recording Sidebar ---
with st.sidebar:
    st.header("🔴 Rec Manager")
    task_instruction = st.text_input("Task Instruction", value="Pick up the red cube")

    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False

    if not st.session_state.is_recording:
        st.button(
            "Start Recording",
            type="primary",
            use_container_width=True,
            on_click=handle_start_recording,
            args=(task_instruction,),
        )
    else:
        col_save, col_discard = st.columns(2)
        with col_save:
            if st.button(
                "✅ Save",
                use_container_width=True,
            ):
                send_stop_recording()
                st.session_state.is_recording = False
                st.rerun()
        with col_discard:
            if st.button(
                "❌ Discard",
                use_container_width=True,
            ):
                send_discard_recording()
                st.session_state.is_recording = False
                st.rerun()
        st.error("RECORDING...")

    st.divider()
    st.header("🔭 Live Physics Monitor")
    phys = st.session_state.physics
    dist = phys.get("target_dist", 0.0)

    # Distance Metric with color cue
    dist_label = "🎯 Target Distance"
    st.metric(
        dist_label,
        f"{dist:.3f} m",
        delta=f"{0.05 - dist:.3f} m" if dist < 0.05 else None,
    )

    # Grasping Status
    is_grasp = phys.get("is_grasping", False)
    if is_grasp:
        st.success("✊ GRASPING: TRUE")
    else:
        st.info("✋ GRASPING: FALSE")

    # Height Metric
    st.metric("📦 Cube Height (Z)", f"{phys.get('cube_z', 0.0):.3f} m")

    lr = st.session_state.lewm_reward
    if lr is not None:
        st.divider()
        st.header("🧠 LeWM Reward Head")
        if "error" in lr:
            st.error(lr["error"])
        else:
            st.metric("reward_pred", f"{lr.get('reward_pred', 0.0):.4f}")
            st.metric(
                "mpc_reward_cost",
                f"{lr.get('mpc_reward_cost', 0.0):.2f}",
                help=lr.get("cost_formula", ""),
            )
            if lr.get("goal_distance") is not None:
                gd_help = lr.get("goal_distance_formula", "")
                if lr.get("goal_distance_mode") == "gallery_latent":
                    gd_help += (
                        f" (goal_id={lr.get('goal_id', 0)}, "
                        f"min over gallery={lr.get('min_gallery_goal_distance', 0):.4f} "
                        f"@ id {lr.get('best_gallery_goal_id', 0)})"
                    )
                elif lr.get("phase_idx") is not None:
                    gd_help += f" (phase_idx={lr['phase_idx']})"
                st.metric(
                    "goal_distance",
                    f"{lr.get('goal_distance', 0.0):.4f}",
                    help=gd_help,
                )
                if lr.get("mpc_goal_cost_term") is not None:
                    st.metric(
                        "mpc_goal_cost_term",
                        f"{lr.get('mpc_goal_cost_term', 0.0):.4f}",
                        help="Semantic term scale used inside MPC get_cost",
                    )
            proxy = lr.get("teleop_progress_proxy")
            if proxy is not None:
                st.metric(
                    "teleop progress proxy",
                    f"{proxy:.4f}",
                    help="(1 - target_dist) * 10 from sim physics",
                )
            hist = lr.get("reward_pred_history")
            if hist:
                st.caption(f"3-frame history: {[f'{x:.3f}' for x in hist]}")
        st.caption(
            "Shown when teleop server is started with "
            "`--query-lewm-reward` (and LeWM serves POST /reward)."
        )

    st.divider()
    st.header("📊 Dataset Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Episodes", st.session_state.total_episodes)
    with col_stat2:
        st.metric("Batch Status", f"{st.session_state.batch_status}/20")

    st.divider()
    st.header("🎯 IK Configuration")
    reach_offset = st.slider("Reach Height (cm)", 5, 40, 5)

    st.header("☁️ Cloud Sync Status")
    if st.session_state.upload_queue > 0:
        st.warning(f"Syncing: {st.session_state.upload_queue} episodes pending...")
    else:
        st.success("✅ All episodes synced to Hub.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Refresh", icon="🔄", use_container_width=True):
            send_command({"command": "poll_status"}, update_lewm_reward=False)
            st.rerun()
    with col_btn2:
        if st.button(
            "Push to Hub", icon="☁️", type="primary", use_container_width=True
        ):
            send_command({"command": "sync"})
            st.rerun()

st.markdown("### Joint Management & State Audit")

# --- Joint and Phase Controls Row ---
col_joint, col_add, col_phase, col_apply = st.columns([3, 1, 3, 1])

with col_joint:
    selected_joint_name = st.selectbox(
        "Pick a joint...",
        options=COMPACT_WIRE_JOINTS,
        index=None,
        placeholder="Select Joint...",
        label_visibility="collapsed",
    )

with col_add:
    if st.button("Add", type="secondary", use_container_width=True):
        if selected_joint_name:
            idx = COMPACT_WIRE_JOINTS.index(selected_joint_name)
            st.session_state.active_joints.add(idx)
            val = st.session_state.staging_buffer[idx]
            if np.isnan(val):
                val = 0.0
                st.session_state.staging_buffer[idx] = val
            if f"input_{idx}" not in st.session_state:
                st.session_state[f"input_{idx}"] = val
            st.rerun()

with col_phase:
    if os.path.exists("../phase_lifecycle.json"):
        with open("../phase_lifecycle.json", "r") as f:
            lifecycle_data = json.load(f)
        phases = sorted(lifecycle_data.keys())
        selected_phase = st.selectbox(
            "Load IK Phase",
            options=["-- Load Phase --"] + phases,
            label_visibility="collapsed",
        )
    else:
        st.info("No lifecycle log.")
        selected_phase = "-- Load Phase --"

with col_apply:
    if st.button("Apply", type="primary", use_container_width=True):
        if selected_phase != "-- Load Phase --" and os.path.exists(
            "../phase_lifecycle.json"
        ):
            with open("../phase_lifecycle.json", "r") as f:
                data = json.load(f)
            apply_snapshot(data[selected_phase])
            st.session_state.last_msg = (f"Loaded {selected_phase}!", "🔭")
            st.rerun()

st.divider()

if not st.session_state.active_joints:
    st.info("No active joints. Select a joint from the dropdown and click 'Add'.")
else:
    for idx in sorted(list(st.session_state.active_joints)):
        name = COMPACT_WIRE_JOINTS[idx]

        # SAFETY INIT: Ensure key exists before rendering slider
        if f"input_{idx}" not in st.session_state:
            st.session_state[f"input_{idx}"] = 0.0

        col_lbl, col_inp, col_clr = st.columns([3, 6, 1])
        with col_lbl:
            st.markdown(f"**[{idx:02}] {name}**")
        with col_inp:
            # SAFETY CLIP: Ensure value from session_state fits in slider bounds
            current_val = float(np.clip(st.session_state[f"input_{idx}"], -1.0, 1.0))
            new_val = st.slider(
                f"Value for {name}",
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                label_visibility="collapsed",
                key=f"input_{idx}",  # Unified key
            )
            st.session_state.staging_buffer[idx] = new_val
        with col_clr:
            if st.button(
                "Remove",
                key=f"remove_btn_{idx}",
                use_container_width=True,
            ):
                st.session_state.active_joints.remove(idx)
                if f"input_{idx}" in st.session_state:
                    del st.session_state[f"input_{idx}"]
                st.session_state.target_buffer[idx] = np.nan
                st.session_state.staging_buffer[idx] = np.nan
                st.rerun()

st.divider()

col_sub, col_reach, col_reset, col_wild, col_snap, col_export, col_clr_all, col_home = (
    st.columns(8)
)
with col_sub:
    st.button(
        "Submit Request",
        type="primary",
        use_container_width=True,
        on_click=handle_submit,
    )

with col_reach:
    st.button(
        "🎯 IK Pickup",
        use_container_width=True,
        on_click=handle_ik_pickup,
        args=(reach_offset,),
    )

with col_reset:
    st.button(
        "Randomize Env",
        use_container_width=True,
        on_click=handle_reset,
    )

with col_wild:
    st.button(
        "Wild Randomize Env",
        use_container_width=True,
        on_click=handle_wild_reset,
    )

with col_snap:
    if st.button("📸 Snapshot", use_container_width=True):
        resp = send_command({"command": "store_snapshot"})
        if resp and resp.get("status") == "snapshot_ok":
            st.toast(f"Snapshot {resp.get('index')} Stored!", icon="📸")
        else:
            st.error("Failed to store snapshot.")

with col_export:
    export_data = {}
    for idx in sorted(list(st.session_state.active_joints)):
        name = COMPACT_WIRE_JOINTS[idx]
        val = st.session_state.staging_buffer[idx]
        export_data[name] = float(val)

    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="Export JSON",
        data=json_str,
        file_name="teleop_settings.json",
        mime="application/json",
        use_container_width=True,
    )

with col_clr_all:
    if st.button(
        "Clear",
        use_container_width=True,
    ):
        clear_all()
        st.rerun()

with col_home:
    if st.button(
        "Home All",
        use_container_width=True,
    ):
        home_all()
        st.rerun()


# --- Phase Machine Executor (Producer) ---
# We execute the next phase at the end of the script so that the CURRENT state
# was rendered for the user before we request the next state and rerun.
if st.session_state.ik_phase is not None and st.session_state.ik_phase < 4:
    phase = st.session_state.ik_phase
    offset = st.session_state.get("ik_offset", 5)
    with st.spinner(f"Executing IK Phase {phase + 1} / 4..."):
        # Small delay to ensure the UI actually shows the state before moving to next
        time.sleep(0.1)
        resp = send_command(
            {"command": "ik_pickup", "phase": phase, "offset_cm": offset}
        )
        if resp and "joints" in resp:
            # Store for the NEXT run to avoid "modified after instantiation" errors
            st.session_state.pending_sync = resp["joints"]
            st.session_state.ik_phase += 1
            if st.session_state.ik_phase >= 4:
                st.session_state.ik_phase = None
                st.session_state.last_msg = (
                    "Pickup Complete! Sliders synced.",
                    "🎯",
                )
            st.rerun()
