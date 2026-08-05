import sys
import os

# Automatically use the virtual environment if the user runs this outside of it
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib')
if os.path.exists(venv_path):
    for root, dirs, files in os.walk(venv_path):
        if 'site-packages' in dirs:
            sys.path.insert(0, os.path.join(root, 'site-packages'))

import torch
import netron
from stable_baselines3 import PPO
from ctde_policy import CTDEPolicy

def main():
    model_path = "ppo_satellite_1.zip"
    onnx_path = "policy.onnx"
    
    print(f"Loading {model_path}...")
    try:
        model = PPO.load(model_path, device="cpu", custom_objects={"policy_class": CTDEPolicy})
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Exporting neural network to ONNX format...")
    # Exporting the MLP extractor is cleaner because it avoids PyTorch distribution sampling ops
    # The extractor takes a single flattened tensor of size 24 (13 local + 11 global)
    dummy_features = torch.randn(1, 24)
    
    # Export the extractor to ONNX format
    torch.onnx.export(
        model.policy.mlp_extractor,
        dummy_features,
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['features_24_dim'],
        output_names=['latent_action', 'latent_value']
    )
    
    print("\\n" + "="*60)
    print("  🌐 LAUNCHING NETRON VISUALIZER 🌐")
    print("="*60)
    print("Open your browser to view the network!")
    print("Press Ctrl+C in this terminal to stop the server.\\n")
    
    # Launch Netron (address can be specified as a tuple if needed, but default is fine)
    netron.start(onnx_path, address=("localhost", 8080), browse=False)
    
    # The netron.start() function spawns a background thread and returns immediately!
    # We must keep the main thread alive so the server doesn't instantly close.
    print("Server is running. Waiting...")
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nShutting down server...")
        netron.stop()

if __name__ == "__main__":
    main()
