### Dynamic Hand Gesture Recognition: Low-Latency Spatiotemporal Classification for Edge Devices

Srikanth Kavoori

> **Key takeaway (for reviewers):** You can control a TV-like interface by waving at a webcam—no remote required. The project built a full pipeline from millions of video frames to a working demo: hand skeletons instead of raw pixels, a simple baseline model, then a tuned **GRU** that recognizes gestures at **85.54%** accuracy with **live inference**. That is strong enough to prove the idea on edge hardware, though slightly below the original **90%** target.

#### Executive summary
This project delivers an efficient, lightweight Human-Computer Interaction (HCI) pipeline for controlling media interfaces on edge devices. The original target was **90%+ accuracy** on 27 Jester gestures with real-time, low-latency inference. By processing the [22.8 GB 20BN-Jester dataset](https://www.kaggle.com/datasets/sanjanatg26/20bn-jester-v1-complete) through Google's MediaPipe framework, raw video was reduced to a **3.24 GB** set of 3D skeletal coordinates. A **Random Forest** baseline (~65% accuracy) confirmed that flattened, non-sequential features miss dynamic motion. A hyperparameter sweep then trained **LSTM and GRU** recurrent models; the winning **GRU** reached **85.54% validation accuracy** and was deployed in a **live webcam demo** ([full report](PROJECT_REPORT.md)).

#### Rationale
Reducing user friction is the ultimate competitive advantage in the smart home and streaming hardware market. Current input methods—like physical remotes or voice commands—can disrupt conversations and frequently fail in noisy environments. By developing a lightweight model deployable on edge hardware, the business gains a silent, immediate, and intuitive interface that boosts user engagement. Furthermore, developing a low-latency spatiotemporal framework for edge devices produces novel technical insights that are highly viable for future developments in the area of Computer Vision, Augemented Reality and Smart home interactions. 

#### Research Question
Can a lightweight spatiotemporal neural network accurately recognize dynamic hand gestures in real-time to control media interfaces, maintaining the low latency required for edge computing devices?

**Answer:** Yes—with **85.54%** validation accuracy, a tuned **GRU** sequence model, and **live webcam inference**, this project shows that lightweight spatiotemporal networks can recognize dynamic hand gestures in real time for HCI-style control on edge-class hardware, though accuracy remained below the original **90%** stretch goal.

#### Data Sources
* **Raw Video Data:** The 20BN-Jester Dataset V1, consisting of over 148,000 short video clips of humans performing diverse, real-world hand gestures.
* **Annotations:** Official Jester ground-truth label mappings (Training and Validation splits) sourced from the Udacity Computer Vision repository.

#### Methodology
1.  **Data Pipeline & Feature Engineering:** Developed a custom Python extraction script utilizing a 64GB RAM M5 Max to pass ~5 million image frames through Google MediaPipe. Extracted 21 ($x, y, z$) hand landmarks per frame, converting dense pixel data into 63 distinct spatial features, significantly reducing the computational load.
2.  **Exploratory Data Analysis (EDA):** Analyzed sequence length distributions to determine an optimal, memory-efficient padding threshold (30 frames). Visualized 3D trajectory paths to confirm spatial patterns existed for distinct classes.
3.  **Baseline Modeling:** A Random Forest Classifier was deployed to establish a non-deep-learning baseline. The 30-frame spatial sequences were flattened into 1D vectors to demonstrate the necessity of a sequence-aware architecture.
4.  **Evaluation Metric:** **Accuracy** was chosen as the primary intuitive metric to gauge the model's overall prediction capability toward our 90% target, supplemented by **Macro F1-Score** to ensure balanced evaluation across all gesture classes.

#### Results
| Stage | Outcome |
|-------|---------|
| **Feature engineering** | MediaPipe reduced raw video to 3D hand landmarks (63 features per frame), isolating movement from background noise. |
| **Exploratory analysis** | Sequence lengths and class balance were profiled; **3D fingertip trajectories** showed visibly different paths across gesture types ([EDA notebook](1_gesture_recognition_eda.ipynb)). |
| **Baseline (Random Forest)** | ~**65%** hold-out accuracy, **~0.64 Macro F1**—strong on static poses, weak on motion-heavy gestures because temporal order is discarded when sequences are flattened. |
| **Deep learning (LSTM vs GRU)** | Grid search over architecture, hidden size, learning rate, and batch size. Best model: **GRU, 256 units, LR 0.0005** → **85.54%** peak validation accuracy at epoch 19 ([tuning notebook](2_gesture_recognition_model_tuning.ipynb)). |
| **Live deployment** | Winning weights power **`live_inference.py`**—webcam → MediaPipe → GRU classification with a confidence filter; demo and setup in [PROJECT_REPORT.md](PROJECT_REPORT.md). |

The **90% accuracy goal was not fully met** on the full 27-class benchmark, but the pipeline is end-to-end: extracted data, validated baseline, tuned sequence model, and shipped real-time inference suitable for smart-home / edge prototypes.

#### Steps Undertaken

* **Exploratory Data Analysis:** CRISP-DM-style EDA, baseline Random Forest with 5-fold CV, and trajectory visualizations — [notebook](1_gesture_recognition_eda.ipynb)
* **Deep Learning & Tuning:** PyTorch **LSTM and GRU** models trained and compared via grid search against the baseline — [notebook](2_gesture_recognition_model_tuning.ipynb)
* **Hardware-Accelerated Training:** Apple Silicon **MPS** on M5 Max for training on the 3.24 GB coordinate dataset
* **Real-Time Deployment:** OpenCV + MediaPipe + GRU live inference prototype — [PROJECT_REPORT.md](PROJECT_REPORT.md)

#### Development environment

To ensure the robustness and reproducibility of the gesture recognition pipeline, I implemented a dual-environment strategy. This decoupling provides two critical benefits:

Dependency Isolation: The Training Environment (.venv_training) is optimized for heavy computational tasks, housing deep learning libraries (PyTorch, Pandas, NumPy) and the hyperparameter tuning suite. The Inference Environment (.venv_inference) is optimized for low-latency, real-time edge performance. By isolating the inference environment, we ensure that the dependencies required for real-time webcam processing (OpenCV, MediaPipe) are stripped of unnecessary training overhead, resulting in a significantly reduced disk footprint and startup time.

Versioning Consistency: The Training Environment is configured for Python 3.9.6 to maintain consistency with the data preprocessing stage where landmarks were generated. The inference environment is specifically built to match the production runtime requirements, preventing 'dependency hell' where training library updates might inadvertently break real-time inference scripts. This modular architecture allows us to deploy the inference script as a standalone, lightweight artifact on edge hardware without needing the full PyTorch training stack.

To achieve this a [setup_env.py](scripts/setup_env.py) script was writen. This script does the following. 

1. Picks a compatible Python 
2. Creates and fills each Virtual Environment. 
3. Smoke Tests each Virtual environments. 
4. Prints information on how to use the environments. 

As an example, below commands help setup the environment needed for running the jupiter notebook for EDA. 

```bash
python3 scripts/setup_env.py
source .venv_training/bin/activate   # Windows: .venv_training\Scripts\activate
jupyter notebook 1_gesture_recognition_eda.ipynb
```

For live inference, activate the inference environment instead: `source .venv_inference/bin/activate`.

`scripts/setup_env.py` creates **`.venv_training`** and **`.venv_inference`**, installing from [`requirements_training.txt`](requirements_training.txt) (notebooks, training, sweep) and [`requirements_inference.txt`](requirements_inference.txt) (webcam demo). Works on macOS, Linux, and Windows. **Python 3.10 or newer** is required (3.11 recommended on Apple Silicon).

#### Outline of project

- [Exploratory Data Analysis (EDA)](1_gesture_recognition_eda.ipynb)
- [Model Evaluation and Fine Tuning](2_gesture_recognition_model_tuning.ipynb)
- [Project Report](PROJECT_REPORT.md)
- [Python Scripts Used for Data Extraction , Model Trainng, Model Evaluation and Live Inference](scripts)
- [Intermediate Models Generated](models)
- [Metrics Collected from Training Models](metrics)
- [Logs Generated during Model Evaluation](logs)
- [Annotations used during Training](annotations)
- [Assets Used in [Jupyter Notebook](2_gesture_recognition_model_tuning.ipynb)](assets)
- [Hand Gesture Co-ordinates Data. Note: This link works only after running the Jupyter Notebook ](data/jester_hand_coordinates.csv)
- [README](README.md)


#### References 

## 6. References & Literature Review
The data pipeline and modeling architecture for this project were heavily informed by recent literature on spatiotemporal classification and edge computing. The following foundational papers were analyzed to guide the dimensionality reduction and LSTM strategies:

1. **S. Kamble, "SLRNet: A Real-Time LSTM-Based Sign Language Recognition System,"** *Department of Artificial Intelligence and Data Science, University of Mumbai, India.*
   * **Relevance:** This paper reinforces the methodology of utilizing Long Short-Term Memory (LSTM) networks for real-time temporal classification. It validates the approach of processing sequential spatial data to achieve low-latency recognition, which is critical for the smart-home edge devices targeted in this project.

2. **M. Madhiarasan and P. P. Roy, "A Comprehensive Review of Sign Language Recognition: Different Types, Modalities, and Datasets,"** *Department of Computer Science and Engineering, Indian Institute of Technology Roorkee.*
   * **Relevance:** This comprehensive review provides foundational context on the various modalities used in gesture recognition. It validates the project's data engineering choice to shift from a dense RGB video modality to a lightweight, skeletal-coordinate modality (via MediaPipe), which isolates spatial features and drastically reduces computational overhead.

3. **E. Uboweja et al., "On-device Real-time Custom Hand Gesture Recognition,"** *Google LLC, Mountain View, CA.*
   * **Relevance:** Authored by researchers behind Google's MediaPipe, this paper directly supports the project's core architecture of utilizing lightweight, on-device spatial tracking. It mathematically validates the capability of running real-time gesture recognition on edge hardware without relying on cloud-based computation, perfectly aligning with the smart-home latency requirements.

4. **M. Oudah, A. Al-Naji, and J. Chahl, "Hand Gesture Recognition Based on Computer Vision: A Review of Techniques,"** J Imaging, 2020.

  * **Relevance:** This comprehensive review establishes that camera vision-based sensors are highly applicable as they provide contactless communication between humans and computers. Furthermore, it mathematically validates your choice of using MediaPipe by confirming that skeleton-based recognition specifies model parameters which can improve the detection of complex features. The paper highlights that skeleton data describes geometric attributes and easily translates features and correlations of data, allowing systems to focus on geometric and statistic features like the skeletal joint location and the space between joints.

##### Contact and Further Information
[Linked In](https://www.linkedin.com/in/srikanthkavoori)