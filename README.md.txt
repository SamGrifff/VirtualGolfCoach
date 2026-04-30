Virtual Golf Coach Artefact

1. Overview

This artefact is the final software implementation for the CS6P05 project "Virtual Golf Coach using Machine Learning and Artificial Intelligence. The system analyses a user-uploaded golf swing video and provides feedback though a Streamlit dashboard. It uses computer vision and pose estimation to extract body landmark data, a pretrained event detection model to identify key swing events and a statistical scoring method based on z-score normalisation to compare the user's swing against a reference template. The application displays an overall swing score, phase-based analysis, deviation charts, key swing frames, biomechanical metrics and coaching feedback. AI-generated feedback is provided using the Gemini API when available. If the API key is not provided or the API fails the system will fallback to predefined rule feedback.

The artefact was testing using Python 3.10.11

2. Implemented Features

The following features have been implemented:


The following features have been implemented:

- Uploading golf swing videos through a Streamlit interface
- Frame extraction and video handling using OpenCV
- Pose estimation using MediaPipe
- Preprocessing, normalisation and resampling of pose data
- Swing event detection using a pretrained deep learning model
- Statistical swing scoring using z-score normalisation
- Phase-based swing analysis
- Biomechanical metrics such as head stability, spine posture, hip rotation and balance
- Frame deviation chart for visualising swing differences
- Key frame display for important swing moments
- AI-generated feedback using the Gemini API
- Rule-based fallback feedback if the Gemini API is unavailable

3. Project Structure

Virtual Golf Coach - Artefact/
│
├── App/
├── data/
├── ml_model/
├── notebooks/
├── swing_event_model/
├── README.md
├── requirements.txt
└── .env.example 

Main folders and files:

App/: Contains the main streamlit application and the main scripts used to run the system.

App/dashboard.py: Main user interface. Allows the user to upload a golf swing video, run the analysis and view the results.

App/video_pipeline.py: Handles video processing, frame extraction, pose estimation, cleaning, normalisation and resampling.

App/event_detector.py: Loads and runs the pretrained golf swing event detection model.

App/analyser.py: Compares processed swing data against the reference template and calculates deviation scores, phase scores and biomechanical metrics.

App/feedback_api: Generates AI feedback using the Gemini API and provides fallback rule-based feedback if AI feedback is unavailable.

data/: Contains sample input data, processed files and the saved reference template needed to test the system.

ml_model/: Contains the reference swing template.

swing_event_model/: Contains the pretrained event detection model and supporting model files.

reuirements.txt: Lists the Python dependencies required to run the project. 

	4. Installation

	Open a terminal in the root folder of the artefact and create a virtual environment using Python 3.10:

		py -3.10 -m venv test_env

	Activate the environment:

		test_env\Scripts\activate

	Upgrade pip:

		python -m pip install --upgrade pip

	Install the required packages:

		pip install -r requirements.txt

5. Running the application

After installing the requirements and activating the virtual environment, run:

streamlit run App/dashboard.py

6. Testing the application

To test the system:

1. Start the Streamlit application.
2. Upload a sample golf swing video.
3. Run the analysis.
4. Check that the system extracts pose data.
5. Check that swing events are detected.
6. Check that an overall score is displayed.
7. Check that phase breakdowns are shown.
8. Check that the deviation chart is generated.
9. Check that feedback is displayed.

If no Gemini API key is provided, the system should still display predefined rule-based feedback.

