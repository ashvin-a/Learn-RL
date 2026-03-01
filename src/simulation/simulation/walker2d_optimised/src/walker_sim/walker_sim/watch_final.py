import mujoco
import mujoco.viewer
import numpy as np
import time
from stable_baselines3 import SAC

def get_observation(data):
    qpos = data.qpos[1:]
    qvel = data.qvel
    return np.concatenate([qpos, qvel]).astype(np.float32)

def main():
    print("Loading Trained Brain...")
    model = SAC.load("sac_walker2d_v1.zip")
    
    print("Loading MuJoCo Body...")
    m = mujoco.MjModel.from_xml_path("walker2d_fixed.xml")
    d = mujoco.MjData(m)

    print("Launching Native Windows GUI...")
    # This opens the native window immediately
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            obs = get_observation(d)
            
            action, _ = model.predict(obs, deterministic=True)
            
            d.ctrl[:] = action
            
            # Step the physics
            mujoco.mj_step(m, d)
            
            viewer.sync()
            
            # Slow it down so it runs in real-time
            time.sleep(0.01)

if __name__ == "__main__":
    main()