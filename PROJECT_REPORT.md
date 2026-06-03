# Final Capstone Report: Real-Time Dynamic Hand Gesture Recognition for HCI

## 1. Executive Summary and Problem Statement
As the tech industry shifts toward spatial computing and smart home interactivity (AR/VR environments, Webcam based Gesture Control on TV), there is an increasing demand for natural Human-Computer Interaction (HCI). The objective of this Capstone project is to develop a robust, real-time dynamic hand gesture recognition system. 

Traditional computer vision approaches rely on skin-color segmentation or heavy frame-by-frame pixel tracking, which are computationally expensive and highly susceptible to variable lighting conditions. This project bypasses those limitations by utilizing a **skeleton-based coordinate approach** leveraging Google's MediaPipe, coupled with Recurrent Neural Networks (RNNs), to classify temporal sequences of hand movements with high accuracy and low latency.


## 2. Literature Review
The architectural and engineering decisions in this project are strictly grounded in recent academic research:

1. **Skeleton-Based Representation:** Oudah et al. (2020) mathematically validated that skeleton-based recognition reduces complex raw RGB video into geometric and statistical features (joint location and spacing), improving the detection of complex temporal features while maintaining contactless HCI.
2. **Computational Modalities:** Madhiarasan & Roy highlighted that extracting skeletal keypoints drastically reduces data dimensionality while preserving necessary spatial information, which is critical for edge-device deployment.
3. **Real-Time Edge Inference:** Uboweja et al. (Google Research) established the foundational framework for MediaPipe, demonstrating that tracking hands in real-time on edge devices without cloud processing is both viable and efficient.
4. **Sequence Modeling:** Kamble's work on *SLRNet* demonstrated the efficacy of feeding sequential spatial data into Long Short-Term Memory (LSTM) networks, perfectly mirroring our approach of using a rolling 30-frame window to capture dynamic gestures.


## 3. Exploratory Data Analysis & Baseline Model Evaluation
The model was trained to identify 27 distinct classes (e.g., "Swiping Left", "Thumb Down") from the 20BN-JESTER dataset. 

Initially, a **Random Forest** baseline model was deployed to establish a benchmark. This required flattening the 30-frame spatial coordinate matrices into 1D arrays (1,890 independent features per sample). 
* **Baseline Result:** The model achieved a Macro F1-Score of 0.64.
* **Failure Analysis:** A review of the classification report revealed that static gestures (e.g., "Thumb Down") achieved high F1-scores (~0.90), while dynamic gestures dependent on trajectory (e.g., "Rolling Hand Backward") failed entirely (~0.45). 
* **Conclusion:** Flattening the arrays destroyed the chronological context. The baseline could not recognize that Frame 2 followed Frame 1. This mathematically justified the transition to a sequence-based Deep Learning architecture capable of maintaining an internal hidden state.


## 4. Deep Learning Methodology & Pipeline
To capture the spatiotemporal dynamics of the gestures, the engineering pipeline was refactored:
* **Feature Extraction (`hand_landmarker_utils.py`):** Utilized the MediaPipe Tasks API to extract $(x, y, z)$ coordinates of hand landmarks using this [script](scripts/hand_landmarker_utils.py)
* **Data Windowing:** Built a `collections.deque` rolling buffer to capture exactly 30 frames of sequential movement, resulting in an input tensor shape of `(1, 30, 63)`.
* **Hardware Acceleration:** Training was accelerated leveraging Apple Silicon Unified Memory on an M5 Max architecture using the PyTorch `mps` backend.


## 5. Sequence Modeling & Hyperparameter Optimization (HPO)
To ensure optimal performance at 30+ FPS during live webcam inference, the network required high accuracy with minimal computational latency. A grid search hyperparameter [swee](scripts/sweep.py) was orchestrated over the following search space in batch sizes of 32 and 64:

* **Architectures:** Long Short-Term Memory (LSTM) vs. Gated Recurrent Unit (GRU). 
* **Hidden Units:** 128 vs. 256.
* **Learning Rates:** 0.001, 0.0005, 0.0001.

*Note: To prevent a combinatorial explosion of training time (which would have exceeded 50 hours), the optimizer was fixed to **Adam**, which is the industry standard for stabilizing RNN training. The computational budget was spent on the most impactful levers: capacity, architecture, and learning rate.*


## 6. Quantitative Results and Final Model Selection
Programmatic analysis of the JSON sweep logs yielded the following definitive winner:

* **Architecture:** Gated Recurrent Unit (GRU)
* **Capacity:** 256 Hidden Units
* **Learning Rate:** 0.0005
* **Convergence:** Epoch 19
* **Peak Validation Accuracy:** 85.54%
* **Validation Loss:** 0.5065

**Architectural Justification:** The sweep proved that the GRU was the superior choice. While LSTMs utilize a complex three-gate mechanism and a distinct cell state, GRUs merge the hidden and cell states using a simplified two-gate mechanism (Update and Reset gates). Because the GRU matched the necessary classification accuracy (85.54%) while requiring significantly fewer tensor operations, it guarantees lower latency and prevents frame-buffering during real-time edge deployment.


## 7. Live Inference Deployment
The winning model weights (`gru_h256_lr0.0005.pth`) were successfully loaded into a standalone, lightweight inference script (`live_inference.py`). The pipeline utilizes OpenCV to capture webcam frames, MediaPipe to extract coordinates, and the GRU to classify the gesture every 30 frames. A confidence threshold of 80% was instituted to filter out transient noise and prevent UI flickering. The demo of the inference functionality can be seen in the video below. Live inference can be run using the following command. 


```
python3 scripts/setup_env.py 
source .venv_inference/bin/activate
python scripts/live_inference.py --model_path gru_h256_lr0.0005.pth --model_type GRU --hidden_size 256
```

[Gesture Recognition Interface Demo](https://youtu.be/o-hSqIJxXEE) demos the custom model. It illustrates the recognition of the following 7 gestures. 
- Thumbs Up
- Thumbs Down 
- Stop Sign 
- Zoom Out with Full Hand 
- Zoom In with Full Hand 
- Zoom Out with Two Fingers 
- Zoom In with Two Fingers


## 8. Bonus Exploration: Advanced Deep Learning HPO
While Grid Search successfully identified the optimal parameters for this project, scaling this system to hundreds of gesture classes would require advanced Deep Learning HPO methodologies. Various Options available for such exploration are

1. **Bayesian Optimization (Optuna/Hyperopt):** Replacing grid search with surrogate probability models that actively predict and test only the most promising hyperparameter combinations, vastly reducing GPU compute time.
2. **Successive Halving (Hyperband):** Allocating small epoch budgets to random configurations and aggressively early-stopping the bottom 50% of performers, reserving compute purely for top-tier models.
3. **Advanced LR Schedulers:** Transitioning from fixed learning rates to dynamic schedulers like **Cosine Annealing with Warm Restarts**, allowing the network to rapidly escape local minima before smoothly settling into the optimal loss landscape.

## 9. Bonus Exploration: Advanced HPO with Optuna
To validate the findings of  exhaustive Grid Search, I conducted a brief exploration using **Optuna**, a hyperparameter optimization framework utilizing Bayesian Optimization (specifically, the Tree-structured Parzen Estimator algorithm).

Unlike grid search, which tests parameters blindly, Optuna built a probability model of the validation accuracy. I allowed it to search a continuous logarithmic space for the learning rate and dynamically choose between LSTM and GRU architectures. Furthermore, I implemented a `MedianPruner`, which aggressively early-stopped (pruned) underperforming models within the first few epochs.

**Findings:** Optuna successfully corroborated the Grid Search results in a fraction of the compute time. The Bayesian optimizer aggressively pruned the heavier LSTM runs due to slower convergence and homed in on the **GRU architecture with 256 hidden units**. It also identified an optimal learning rate natively within the `~0.0005` range. The exact results are shown below. This exercise demonstrated that while Grid Search is excellent for discrete mapping, Bayesian methods with early-stopping are vastly superior for scaling deep learning pipelines. Further while it took 18 hours for the grid search, Optuna was able to find the best model in 4.5 hours reducing the time of tuning by a factor of 4. 

```
==================================================
🏆 OPTUNA BAYESIAN OPTIMIZATION RESULTS 🏆
==================================================
Best Trial Score: 0.8505
Best Hyperparameters:
    model_type: GRU
    hidden_size: 256
    lr: 0.00028076976159655274
```

---

## 10. Conclusion
Over the last two months working on this Capstone project, I was able to successfully explore the end-to-end lifecycle of training, evaluating, tuning, and deploying a Deep Learning Neural Network for a real-world Human-Computer Interaction (HCI) application. 

This project spanned a wide variety of technologies, utilized advanced hardware acceleration, and integrated an array of tools to build a production-grade machine learning pipeline. Key milestones included:

1. **Computer Vision & Feature Extraction:** Moving beyond heavy RGB video tracking by utilizing Google's MediaPipe Holistic to extract lightweight, 3D skeletal $(x, y, z)$ coordinates.
2. **Spatiotemporal Deep Learning:** Transitioning from baseline machine learning (Random Forests) to Recurrent Neural Networks (GRUs/LSTMs) capable of understanding chronological sequences and complex movement trajectories.
3. **Hardware & Optimization:** Accelerating matrix multiplications using Apple Silicon Unified Memory (MPS backend) and executing rigorous Hyperparameter Tuning to balance classification accuracy against edge-device inference latency.
4. **Live Deployment:** Engineering a multi-threaded Python pipeline using OpenCV and `collections.deque` rolling buffers to successfully classify hand gestures in real-time via a live webcam feed.

Ultimately, this project proves the viability of using efficient, skeleton-based sequence models to power natural, contactless interfaces for smart-home and spatial computing environments.

