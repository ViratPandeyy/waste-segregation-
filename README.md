# Smart AI Waste Segregation — Mac edition

## What is included
- Streamlit web app with live camera classification
- MobileNetV2 training script for four categories: plastic, glass, paper, organic
- Mac setup and launch command files

**Important:** The ZIP does not contain a trained model or sample training dataset. Add your own correctly labelled images, train the model, then launch the app. Phase 1 classifies the centre region of the camera view; it is not multi-object bounding-box detection.

## Requirements
- macOS
- Python 3.11
- Internet connection for package installation and the first MobileNetV2 ImageNet weights download

## Setup
1. Extract this ZIP.
2. Open the extracted folder in Finder.
3. Add training images into `dataset/plastic`, `dataset/glass`, `dataset/paper`, and `dataset/organic` (aim for at least 100 varied images per class; more is better).
4. Control-click `setup_mac.command`, choose **Open**, and follow prompts. Or run `./setup_mac.command` from Terminal.
5. In Terminal, activate the environment and train:

```bash
source .venv/bin/activate
python train.py --epochs 12
```

6. Start the website by double-clicking `run_mac.command` (Control-click > Open if macOS blocks it), or run:

```bash
source .venv/bin/activate
streamlit run app.py
```

7. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Camera permission
If prompted, allow camera access for the browser. If blocked, open **System Settings → Privacy & Security → Camera**, enable access for your browser, then restart the browser and app.

## Notes
- For a camera feed from another device, the app must be hosted on a network-accessible server and served over HTTPS; localhost only works on the Mac itself.
- TensorFlow compatibility can vary by Mac chip and macOS version. If package installation fails, share the exact error and whether the Mac is Apple Silicon (M1/M2/M3/M4) or Intel.
