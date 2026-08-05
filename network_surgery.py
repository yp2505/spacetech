import torch
import zipfile
import os
import shutil
import tempfile

def pad_tensor(old_tensor, new_shape):
    """
    Takes an old tensor (e.g., 64x64) and pads it with zeroes to match new_shape (e.g., 128x128).
    The old weights are placed in the top-left corner.
    """
    new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype, device=old_tensor.device)
    
    if len(old_tensor.shape) == 1:
        # Bias vector
        new_tensor[:old_tensor.shape[0]] = old_tensor
    elif len(old_tensor.shape) == 2:
        # Weight matrix
        new_tensor[:old_tensor.shape[0], :old_tensor.shape[1]] = old_tensor
    else:
        raise ValueError(f"Unsupported tensor dimension: {len(old_tensor.shape)}")
        
    return new_tensor

def perform_surgery(zip_path, output_path, new_actor_hidden, new_critic_hidden):
    print(f"Starting Network Surgery on {zip_path}...")
    
    # 1. Extract the zip file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    # 2. Load the old weights (policy.pth)
    policy_path = os.path.join(temp_dir, "policy.pth")
    old_state_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
    
    # We will build a new state dict
    new_state_dict = {}
    
    for name, old_tensor in old_state_dict.items():
        # Determine new shape based on the layer name
        new_shape = list(old_tensor.shape)
        
        # ACTOR NETWORKS (was 128 -> 64, now 128 -> 128)
        if "mlp_extractor.actor_net.0.weight" in name: # Linear(13, 128) -> Linear(13, 128) [No change for first layer output if new arch is 128, but wait old was 128. Oh, old was [128, 64]]
            # Old: (128, 13). New: (128, 13). No change in shape!
            pass
        elif "mlp_extractor.actor_net.0.bias" in name:
            pass
        elif "mlp_extractor.actor_net.2.weight" in name: # Linear(128, 64) -> Linear(128, 128)
            # Old: (64, 128). New: (128, 128)
            new_shape[0] = 128
        elif "mlp_extractor.actor_net.2.bias" in name:
            new_shape[0] = 128
        elif "action_net.weight" in name: # Linear(64, 3) -> Linear(128, 3)
            # Old: (3, 64). New: (3, 128)
            new_shape[1] = 128
            
        # CRITIC NETWORKS (was [256, 128, 64], now [256, 256, 128])
        elif "mlp_extractor.critic_net.0.weight" in name: # Linear(11, 256) -> Linear(11, 256)
            pass
        elif "mlp_extractor.critic_net.0.bias" in name:
            pass
        elif "mlp_extractor.critic_net.2.weight" in name: # Linear(256, 128) -> Linear(256, 256)
            # Old: (128, 256). New: (256, 256)
            new_shape[0] = 256
        elif "mlp_extractor.critic_net.2.bias" in name:
            new_shape[0] = 256
        elif "mlp_extractor.critic_net.4.weight" in name: # Linear(128, 64) -> Linear(256, 128)
            # Old: (64, 128). New: (128, 256)
            new_shape[0] = 128
            new_shape[1] = 256
        elif "mlp_extractor.critic_net.4.bias" in name:
            new_shape[0] = 128
        elif "value_net.weight" in name: # Linear(64, 1) -> Linear(128, 1)
            # Old: (1, 64). New: (1, 128)
            new_shape[1] = 128
            
        # Perform padding if the shape has grown
        if list(old_tensor.shape) != new_shape:
            print(f"Padding {name} from {list(old_tensor.shape)} to {new_shape}")
            new_state_dict[name] = pad_tensor(old_tensor, tuple(new_shape))
        else:
            new_state_dict[name] = old_tensor.clone()
            
    # 3. Save the new weights back to policy.pth
    torch.save(new_state_dict, policy_path)
    
    # 4. Repackage the zip file
    print(f"Repackaging into {output_path}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
                
    shutil.rmtree(temp_dir)
    print("Surgery complete! The brain has been successfully expanded.")

if __name__ == "__main__":
    # We apply surgery to both satellites
    perform_surgery("ppo_satellite_1.zip", "ppo_satellite_1_upgraded.zip", [128, 128], [256, 256, 128])
    perform_surgery("ppo_satellite_2.zip", "ppo_satellite_2_upgraded.zip", [128, 128], [256, 256, 128])
    
    # Replace the old files so train.py loads them directly
    shutil.move("ppo_satellite_1_upgraded.zip", "ppo_satellite_1.zip")
    shutil.move("ppo_satellite_2_upgraded.zip", "ppo_satellite_2.zip")
    print("Files overwritten. Ready to run train.py!")
