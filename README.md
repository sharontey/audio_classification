# audio_classification
Create an audio classification model that processes environmental audio.
# Setup

Make sure you have the ESC-50 folder with all of its contents and the esc50.csv in the root directory. You can download the ESC-50 dataset here: https://github.com/karolpiczak/ESC-50

# How to run

1) Make sure Anaconda is installed. If you run "conda" in the terminal and get some output, then you are good
2) Setup the environment. Run the following on the terminal, make sure you are in the audio_classification directory:
- conda env create -f environment.yml
- conda activate audio_classification
- python -m ipykernel install --user --name audio_classification --display-name "Python 3.12
  (audio_classification)"
3) Select "Python 3.12 (audio_classification)" as the kernel in Jupyter
