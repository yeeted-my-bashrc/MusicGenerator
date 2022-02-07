# MusicGenerator
MusicGenerator for GunnHacks 8.0 Hackathon
## Running
First, download the repository
```
git clone --recurse-submodules https://github.com/yeeted-my-bashrc/MusicGenerator.git
```
Then enter the repository folder
```bash
cd MusicGenerator
```
Initialize a venv and install the dependencies
```bash
python -m venv .venv
# on unix
source .venv/bin/activate
# on windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
cd music21
python -m pip install .
```
To train the AI use
```
python _train.py
```
To have it generate music use
```
python generate.py
```