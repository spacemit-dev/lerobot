# LeRobot User Guide

This document describes how to use **LeRobot** on the **K1** platform. It provides a complete, end-to-end (E2E) workflow for running robotic arm applications on K1, including:

- Calibration, teleoperation, and data collection for the **SO-101 dual-arm robot** (leader + follower)
- Training an **ACT (Action Chunking Transformer)** model on an x86 workstation and deploying it to K1 for local inference to perform a cube pick-and-place task
- Fine-tuning a **SmolVLA** model on an x86 workstation and deploying it in a distributed setup to complete the same task

## Hardware and Software Requirements

### Hardware

The following hardware is required:

- **Training machine**: x86 workstation with an NVIDIA RTX-class GPU or better
- **Target device**: K1 development board with **Bianbu ROS firmware**
- **Robot**: SO-101 leader–follower robotic arm
- **Vision input**: Two USB cameras

## Software Environment Setup

> **Note**
> Two environments are required:
>
> - **x86 development environment**: Used for visualized data collection and model training
> - **K1 local environment**: Used for model deployment and inference
>
> Lerobot SDK must be installed on **both** the x86 workstation and the K1 board.
> Unless otherwise specified, commands are executed on **K1**.

<!--

### Download the Source Code

```bash
wget https://archive.spacemit.com/ros2/prebuilt/brdk_integration/python/lerobot/lerobot.tar.gz
tar -xvf lerobot.tar.zg -C ~
```

> **Caution**
>
> To reduce total bandwidth when streaming multi-view video, **Bianbu ROS converts video frames to MJPG format** based on the official LeRobot source code.
> No other functional changes were made to ensure maximum upstream compatibility.

-->

### Install System Dependencies

Update the system and install required packages:

```bash
sudo apt update
sudo apt install python3-venv ffmpeg
```

- **python3-venv**: Used to manage pip virtual environments
- **ffmpeg**: Required for video frame processing

### Install Python Dependencies

<!--

**Option 1:** Use the Prebuilt Virtual Environment

```bash
wget https://archive.spacemit.com/ros2/prebuilt/brdk_integration/python/lerobot/lerobot-venv.tar.gz
tar -xvf lerobot-venv.tar.gz -C ~
source ~/.lerobot-venv/bin/activate
```

**Option 2:** Install from LeRobot Source

-->

```bash
python3 -m venv ~/.lerobot-venv
source ~/.lerobot-venv/bin/activate
cd ~/lerobot
pip install -e . && pip install "lerobot[all]"
```

## Pre-Deployment Setup

### Robotic Arm Calibration

1. Before calibration, ensure the robot is fully assembled and motors are configured with:

   - [SO-101 assembly guide](https://huggingface.co/docs/lerobot/so101#step-by-step-assembly-instructions)
   - [Motor configuration](https://huggingface.co/docs/lerobot/so101#configure-the-motors)

   Then, Connect both arms (power + USB) and run:

   ```bash
   lerobot-find-port
   ```

2. Grant serial port permission
  
   USB UART devices typically appear as `/dev/ttyACM*` on K1:

   ```Bash
   sudo chmod 666 /dev/ttyACM0
   ```

3. Calibrate each arm

   ```Bash
   # Follower arm
    lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm # 自定义
   
   # Leader arm
    lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm # 自定义
   ```

   **Note:** Make sure the device names match those on your system. For the detailed procedure, refer to the [Hugging Face official calibration guide](https://huggingface.co/docs/lerobot/so101#calibration-video) for details.

### Teleoperation

1. **Verify Robot Ports**

   Before starting teleoperation or data collection, verify the robotic arm’s serial port using the command below.

   ```Bash
   lerobot-find-port
   ```

2. **Teleoperation Without Cameras**

   Once the serial ports are correctly configured, run the following command to start teleoperation without cameras.

   ```Bash
   lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm
   ```

3. **Camera Setup and Validation**

   Two USB cameras are recommended:
   - **Top view**: Mounted above the workspace for a global view
   - **Side view**: Mounted on the side for fine-grained manipulation details

   The camera placement follows these principles:
   - Key details of the task execution must be clearly captured
   - The field of view should exclude unrelated objects
   - Camera positions must remain fixed to ensure dataset quality and accuracy

   After fixing the camera viewpoints, connect both USB cameras to the **K1 development board**, and run the following command to check the assigned camera IDs.

   ```bash
   lerobot-find-cameras opencv
   ```

   Example output from terminal:

   ```Bash
   --- Detected Cameras ---
   Camera #0:
    Name: OpenCV Camera @ /dev/video2
    Type: OpenCV
    Id: /dev/video20
    Backend api: V4L2
    Default stream profile:
      Format: 0.0
      Width: 640
      Height: 480
      Fps: 30.0
   --------------------
   Camera #1:
    Name: OpenCV Camera @ /dev/video4
    Type: OpenCV
    Id: /dev/video22
    Backend api: V4L2
    Default stream profile:
      Format: 0.0
      Width: 640
      Height: 480
      Fps: 30.0
   --------------------

   Finalizing image saving...
   Image capture finished. Images saved to outputs/captured_images
   ```

   Locate the images captured by each camera in the `outputs/capture_images` directory, and verify the port ID corresponding to each camera position.

4. **Visualized Teleoperation**

   After confirming camera IDs, run the following command to verify camera quality and framing.

   ```Bash
   lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{
        top:  {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
   ```

### Dataset Collection

1. Before starting data collection, you may choose whether to log in to `huggingface-cli`.
   Logging in allows you to conveniently upload datasets and models to the Hugging Face Hub.

   ```Bash
   hf auth login
   ```

   Follow the prompts to enter your Hugging Face access token.

2. After logging in, you can obtain and set `<HF_USER>` as follows:

   ```Bash
   HF_USER=$(hf auth whoami | head -n 1 | awk '{print $3}')
   echo $HF_USER
   ```

   If `<HF_USER>` is not specified, you must manually replace `<HF_USER>` in the following content with an arbitrary name.

3. Start the data collection

   Run the following command to begin data collection:

   ```Bash
   lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{
        top:  {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
    --dataset.repo_id=${HF_USER}/record-green-cube \
    --dataset.single_task="Place the green cube into the box" \
    --dataset.root=./datasets/record-green-cube \
    --dataset.push_to_hub=True \
    --play_sounds=false \
    --display_data=true # This is recommended on the x86 workstation
   ```

   - **Parameter Descriptions**

     - `dataset.num_episodes`: Specifies the expected number of data episodes to be collected
     - `dataset.episode_time_s`: Specifies the duration of each data collection episode (in seconds)
     - `dataset.reset_time_s`: Preparation time between consecutive data collection episodes (in seconds)
     - `dataset.repo_id`:
       - `$HF_USER`: the current user
       - `record-green-cube`: the dataset name
     - `dataset.single_task`: Task instruction, which can be used as input for VLA models
     - `dataset.root`: Specifies the dataset storage location; defaults to `~/.cache/huggingface/lerobot/`
     - `dataset.push_to_hub`: Determines whether to upload the dataset to the Hugging Face Hub
     - `play_sounds`：Enables or disables instruction audio playback
     - `display_data`：Enables or disables the graphical interface. **If enabled, data collection is recommended on an x86 server**

     For detailed command usage, run the command with `--help`.

- **Checkpointing and Recovery**
  - Checkpoints are created automatically during recording
  - Resume recording with `--resume=true` for any issues
  - To restart from scratch, **manually delete** the dataset directory

- **Keyboard controls during recording (X11 mode only)**
  - Press **Right Arrow (**`→`**)**: End the current episode early, or reset the timer and move to the next episode
  - Press **Left Arrow (**`←`**)**: Discard the current episode and re-record it
  - Press **Esc (**`ESC`**)**: Immediately stop the session, encode the videos, and upload the dataset

- **Tips of record**

  - Start with a simple task, such as picking up objects from different positions and placing them into a bin
  - Aim to record at least **50 episodes** in total, with **around 10 episodes for each position**
  - Keep the cameras fixed throughout the entire recording process, and perform the grasping actions in a consistent manner
  - Make sure the task can be completed **by relying only on the camera images**, without relying on external cues

## ACT Model Training and Deployment

ACT (Action Chunking Transformer) is an imitation learning algorithm proposed by the ALOHA team in April 2023. It is designed to address the challenges of fine-grained manipulation tasks.

ACT combines the strong representation capability of Transformer models with action chunking techniques, enabling it to learn more complex action sequences for tasks such as robot control, while maintaining efficient execution over long-horizon tasks.

### Model Training (x86 workstation)

**Move the Dataset to the workstation**

If the dataset was collected locally on the K1 device, it must be transferred to the development machine before training:

1. If the workstation is properly configured with a network proxy and the dataset `push` has already been **pushed to Hugging Face**, you can load the dataset during training via `repo_id`.
   In this case, make sure to log in `huggingface` from the station terminal.

2. If no proxy is available and the dataset is stored locally, manually move the dataset to the following directory on the workstation:
   `~/lerobot/datasets`

**wandb Setup**

Optionally enable **wandb** to monitor training metrics and loss curves.

Run the following command to log in:
```Bash
wandb login
```

**Model Trainin**

Run the following command to start training.
Training parameters can be adjusted in the configuration file:
`src/lerobot/configs/train.py`

```Bash
lerobot-train \
  --dataset.repo_id={HF_USER}/record-green-cub \
  --dataset.root=datasets/record-green-cube \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_pickplace \
  --job_name=act_so101_pickplace \
  --policy.device=cuda \
  --steps=200000
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_act_policy
```

- **Parameter Descriptions**
  - `dataset.repo_id`: Downloads the dataset from Hugging Face for training
  - `dataset.root`: Uses a local dataset for training (takes priority over `dataset.repo_id`)
  - `policy.type`: Policy type to train from scratch
  - `output_dir`: Directory for saving model checkpoints and wandb logs
  - `job_name`: Name of the training job
  - `policy.device`: Training device (`cpu` | `cuda` | `mps`)
  - `wandb.enable`: Enables or disables wandb logging
  - `policy.repo_id`: Repository ID for saving the trained policy

> **Note**
> If the trained the model using an **RTX 4090 GPU**, with a dataset of **60 episodes**.
> Training was run for **200k steps** (until the loss converged), with a **batch size of 8**, and took approximately **4 hours**.

### Model Deployment

**Copy the Model to the K1 Development Board**

After completing model training on the development machine, copy the final model checkpoint to the `lerobot` directory on the **K1 development board**.
The model path and directory structure are shown below:

```bash
(.lerobot-venv) ➜  lerobot git:(main) ✗ tree outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model
outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model
├── config.json
├── model.safetensors
└── train_config.json

1 directory, 3 files
```

- `config.json`: Model configuration file, including hyperparameters and other settings
- `model.safetensors`: File containing the trained model weights
- `train_config.json`: Configuration used during training, recording the training parameters

**Model Inference**

Once the model is deployed to the K1 development board, you can run inference locally.
The following example shows how to execute a **grasping task** using the trained model:

```bash
lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{
        top:  {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
    }" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_act \
  --dataset.single_task="Place the greeb cube into the box" \
  --policy.path=outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model \
  --policy.device=cpu \
  --dataset.episode_time_s=180 \
  --dataset.reset_time_s=30 \
  --play_sounds=false
```

## SmolVLA Model Fine-tuning and Inference

**SmolVLA** is a lightweight **Vision–Language–Action (VLA)** model with approximately **450M parameters**, enabling efficient training and deployment on **consumer-grade GPUs**.

Built on top of a **Vision–Language Model (VLM)**, SmolVLA integrates an **Action Expert** module that allows the model to understand visual inputs (such as images or video streams) together with natural language instructions, and generate corresponding **robot action sequences**.

![image](images/smolvla.png)

### Model Fine-tuning (x86 Workstation)

It is recommended to fine-tune SmolVLA based on the **official base model released on Hugging Face**.
Starting from a pretrained base model allows the system to quickly adapt general visual and language understanding to **task-specific scenarios**.

Fine-tuning Command as below:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/record-green-cub \
  --dataset.root=datasets/record-green-cube \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=${HF_USER}/my_smolvla_policy_findtune \
  --output_dir=outputs/train/smolvla_so101_pickplace_finetune \
  --job_name=smolvla_so101_pickplace \
  --policy.device=cuda \
  --steps=200000 \
  --wandb.enable=true
```

- `policy.path`: Path or name of the base model.
  In this example, it refers to the **`smolvla_base` model provided by the LeRobot project**.

### Distributed Deployment

The compute capability of the K1 device is not sufficient to run the **SmolVLA** model locally.
Instead, SmolVLA can be deployed using the **distributed deployment approach provided by the LeRobot project**.

In this setup:

- The **K1 device acts as the client**, responsible for data collection and action execution
- The **x86 workstation acts as the server**, running the SmolVLA model and performing inference to generate action sequences
- The client and server communicate via the **gRPC protocol**

During operation, the client sends captured visual data and sensor observations to the server. The server performs inference using the SmolVLA model, generates the corresponding action sequences, and sends them back to the client, which then drives the robotic arm to execute the grasping task.

Server command as below:

```
python src/lerobot/scripts/server/policy_server.py \
     --host=0.0.0.0 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```

Client command as below:

```bash
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{
        top:  {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
        side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
    }" \
    --robot.id=my_awesome_follower_arm \
    --task="Place the greeb cube into the box" \
    --server_address=${server_ip}:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=outputs/train/smolvla_so101_pickplace_finetune/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

## FAQ

### Rerun Rendering Failure

On **Bianbu ROS**, the graphics rendering backend is **OpenGL ES (GLES)**.
By default, **Rerun** uses **Vulkan** as its rendering backend, which may cause rendering failures.

Use the following environment variable to explicitly select the **GLES** backend:

```Bash
export WGPU_BACKEND=gles
```
