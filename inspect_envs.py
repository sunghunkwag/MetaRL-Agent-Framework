import gymnasium as gym
import sys

print("Checking imports...")
try:
    import metaworld
    print("MetaWorld imported successfully")
except ImportError as e:
    print(f"MetaWorld import failed: {e}")

try:
    import imujoco
    print("iMuJoCo imported successfully")
except ImportError as e:
    print(f"iMuJoCo import failed: {e}")

print("\nChecking Gym Registry for 'iMuJoCo'...")
try:
    all_envs = list(gym.envs.registry.keys())
    imujoco_envs = [id for id in all_envs if 'iMuJoCo' in id or 'imujoco' in id]
    print(f"Found {len(imujoco_envs)} iMuJoCo envs:")
    for env_id in imujoco_envs[:10]:
        print(f"  - {env_id}")
except Exception as e:
    print(f"Error listing envs: {e}")
