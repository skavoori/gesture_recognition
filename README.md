### Project Title
#### Dynamic Hand Gesture Recognition: Low-Latency Spatiotemporal Classification for Edge Devices

**Author**
Srikanth Kavoori

#### Executive summary
This project explores an efficient, lightweight approach to Human-Computer Interaction (HCI) to control media interfaces on edge computing devices. The goal is to develop a model capable of classifying core smart-home control gestures (e.g., swipe, palm, pinch) with over 90% accuracy while maintaining real-time, low-latency performance. By processing the massive [22.8 GB 20BN-Jester video dataset](https://www.kaggle.com/datasets/sanjanatg26/20bn-jester-v1-complete) through Google's MediaPipe framework, raw video pixel data was condensed into a 3.24 GB dataset of purely mathematical 3D skeletal coordinates. To establish a performance baseline, a Random Forest Classifier was evaluated on the flattened coordinate vectors. The inherent limitations of this non-sequential model in capturing dynamic movement strongly validate the next phase of this project: ***Designing a Recurrent Neural Network (LSTM) engineered explicitly for edge-based spatiotemporal classification.***

#### Rationale
Reducing user friction is the ultimate competitive advantage in the smart home and streaming hardware market. Current input methods—like physical remotes or voice commands—can disrupt conversations and frequently fail in noisy environments. By developing a lightweight model deployable on edge hardware, the business gains a silent, immediate, and intuitive interface that boosts user engagement. Furthermore, developing a low-latency spatiotemporal framework for edge devices produces novel technical insights that are highly viable for future developments in the area of Computer Vision, Augemented Reality and Smart home interactions. 

#### Research Question
Can a lightweight spatiotemporal neural network accurately recognize dynamic hand gestures in real-time to control media interfaces, maintaining the low latency required for edge computing devices?

#### Data Sources
* **Raw Video Data:** The 20BN-Jester Dataset V1, consisting of over 148,000 short video clips of humans performing diverse, real-world hand gestures.
* **Annotations:** Official Jester ground-truth label mappings (Training and Validation splits) sourced from the Udacity Computer Vision repository.

#### Methodology
1.  **Data Pipeline & Feature Engineering:** Developed a custom Python extraction script utilizing a 64GB RAM M5 Max to pass ~5 million image frames through Google MediaPipe. Extracted 21 ($x, y, z$) hand landmarks per frame, converting dense pixel data into 63 distinct spatial features, significantly reducing the computational load.
2.  **Exploratory Data Analysis (EDA):** Analyzed sequence length distributions to determine an optimal, memory-efficient padding threshold (30 frames). Visualized 3D trajectory paths to confirm spatial patterns existed for distinct classes.
3.  **Baseline Modeling:** A Random Forest Classifier was deployed to establish a non-deep-learning baseline. The 30-frame spatial sequences were flattened into 1D vectors to demonstrate the necessity of a sequence-aware architecture.
4.  **Evaluation Metric:** **Accuracy** was chosen as the primary intuitive metric to gauge the model's overall prediction capability toward our 90% target, supplemented by **Macro F1-Score** to ensure balanced evaluation across all gesture classes.

#### Results
The dimensionality reduction strategy was highly successful, isolating the core signal (hand movement) from background video noise. The Exploratory Data Analysis confirmed that distinct gestures produce unique mathematical 3D trajectories. However, the baseline Random Forest model struggled to achieve high accuracy. Because the Random Forest treats the flattened spatial coordinates as independent variables, it fails to capture the chronological sequence of the gesture—mathematically proving that the temporal order of the frames is the most critical signal for classification. 

#### Next steps
* **Deep Learning Architecture:** Construct a PyTorch Long Short-Term Memory (LSTM) network. Unlike the baseline, the LSTM's internal hidden state is designed specifically to track chronological spatiotemporal movement patterns over time.
* **Hardware Accelerated Training:** Utilize Apple Silicon Metal Performance Shaders (`mps`) on the M5 Max to train the model efficiently on the 3.24 GB coordinate dataset to push beyond the 90% accuracy threshold.
* **Real-Time Deployment:** Develop a live OpenCV webcam pipeline to feed real-time $(x, y, z)$ coordinates into the trained model for live, on-device gesture classification to simulate the edge computing media interface.

#### Development environment

Do not commit a virtual environment to git (it is platform-specific and large). Instead, from the project root, run:

```bash
python3 scripts/setup_env.py
source .venv/bin/activate   # Windows: .venv\Scripts\activate
jupyter notebook gesture_recognition_eda.ipynb
```

`scripts/setup_env.py` creates `.venv/`, installs packages from `requirements.txt` (including Jupyter), and works on macOS, Linux, and Windows. Python 3.10 or newer is required.

#### Outline of project

- [Exploratory Data Analysis (EDA)](gesture_recognition_eda.ipynb)
- [Environment setup](scripts/setup_env.py)
- [Script used for data processing](scripts/extract_landmarks.py)
- [Annotations](annotations)
- [Hand Gesture Co-ordinates Data. Note: This link works only running the Jupyter Notebook](data/jester_hand_coordinates.csv)
- [Assets Used in Jupyter Notebook](assets)
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