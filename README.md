# NFL Game Prediction Project
## Current Record: 12 - 4
This project builds and trains a machine learning model to predict the outcome of NFL games using historical data, play-by-play stats, betting lines, and injuries.

## How it works
- **train.py** → trains models (Logistic Regression, Random Forest, XGBoost) and saves the best one.  
- **predict.py** → generates win probabilities for upcoming games.  
- **features.py** → builds features from games, play-by-play, and injuries.  
- **utils.py** / **config.py** → helper functions and settings.  
- **db.py** → handles database connections.  

Some loader scripts (e.g. for scraping injuries and schedules) are excluded to keep the repo clean and because they require private data sources.

## Install
Clone the repo and install requirements:
```bash
pip install -r requirements.txt
