# Virtual Golf Coach Artefact

## 1. Overview

This artefact is the final software implementation for the CS6P05 project **"Virtual Golf Coach using Machine Learning and Artificial Intelligence"**.

The system analyses a user-uploaded golf swing video and provides feedback through a Streamlit dashboard. It uses computer vision and pose estimation to extract body landmark data, a pretrained event detection model to identify key swing events, and a statistical scoring method based on z-score normalisation to compare the user's swing against a reference template.

The application displays:

- Overall swing score
- Phase-based swing analysis
- Deviation charts
- Key swing frames
- Biomechanical metrics
- Coaching feedback

AI-generated feedback is provided using the Gemini API when available. If the API key is not provided, or if the API fails, the system falls back to predefined rule-based feedback.

This artefact was tested using **Python 3.10.11**.

---

## 2. Implemented Features

The following features have been implemented:

- Uploading golf swing videos through a Streamlit interface
- Frame extraction and video handling using OpenCV
- Pose estimation using MediaPipe
- Preprocessing, normalisation and resampling of pose data
- Swing event detection using a pretrained deep learning model
- Statistical swing scoring using z-score normalisation
- Phase-based swing analysis
- Biomechanical metrics including head stability, spine posture, hip rotation and balance
- Frame deviation charts for visualising swing differences
- Key frame display for important swing moments
- AI-generated feedback using the Gemini API
- Rule-based fallback feedback if the Gemini API is unavailable

---

## 3. Project Structure

```text
Virtual Golf Coach - Artefact/
│
├── App/
│   ├── dashboard.py
│   ├── video_pipeline.py
│   ├── event_detector.py
│   ├── analyser.py
│   └── feedback_api.py
│
├── data/
├── ml_model/
├── notebooks/
├── swing_event_model/
├── README.md
├── requirements.txt
└── .env.example
```

### Main Folders and Files

| File / Folder | Description |
|---|---|
| `App/` | Contains the main Streamlit application and scripts used to run the system. |
| `App/dashboard.py` | Main user interface. Allows the user to upload a golf swing video, run the analysis and view the results. |
| `App/video_pipeline.py` | Handles video processing, frame extraction, pose estimation, cleaning, normalisation and resampling. |
| `App/event_detector.py` | Loads and runs the pretrained golf swing event detection model. |
| `App/analyser.py` | Compares processed swing data against the reference template and calculates deviation scores, phase scores and biomechanical metrics. |
| `App/feedback_api.py` | Generates AI feedback using the Gemini API and provides fallback rule-based feedback if AI feedback is unavailable. |
| `data/` | Contains sample input data, processed files and saved files needed to test the system. |
| `ml_model/` | Contains the reference swing template. |
| `swing_event_model/` | Contains the pretrained event detection model and supporting model files. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `.env.example` | Example environment file for setting up the Gemini API key. |

---

## 4. Installation

Open a terminal in the root folder of the artefact.

### Step 1: Create a virtual environment

```bash
py -3.10 -m venv test_env
```

### Step 2: Activate the virtual environment

```bash
test_env\Scripts\activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install the required packages

```bash
pip install -r requirements.txt
```

---

## 5. Gemini API Setup

The system can generate AI feedback using the Gemini API.

Create a `.env` file in the root folder and add your API key:

```env
GEMINI_API_KEY=your_api_key_here
```

If no Gemini API key is provided, the system will still run and display predefined rule-based feedback.


---

## 6. Running the Application

After installing the requirements and activating the virtual environment, run:

```bash
streamlit run App/dashboard.py
```

This will open the Streamlit dashboard in your browser.

---

## 7. Testing the Application

To test the system:

1. Start the Streamlit application.
2. Upload a sample golf swing video.
3. Run the analysis.
4. Check that the system extracts pose data.
5. Check that swing events are detected.
6. Check that an overall score is displayed.
7. Check that phase breakdowns are shown.
8. Check that the deviation chart is generated.
9. Check that coaching feedback is displayed.

If no Gemini API key is provided, the system should still display predefined rule-based feedback.

---

## 8. Requirements

The project uses the following main libraries:

- Streamlit
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- PyTorch
- Torchvision
- python-dotenv
- google-genai

Install all dependencies using:

```bash
pip install -r requirements.txt
```
