import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import joblib
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class NFLPlayPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.X_data = None
        self.y_data = None
        self.formation_map = {'S': 'SHOTGUN', 'U': 'UNDER CENTER', 'P': 'PISTOL', 'N': 'NO HUDDLE SHOTGUN', 'H': 'NO HUDDLE'}

    def load_data(self):
        logging.info("Loading data...")
        try:
            plays = pd.read_csv('plays.csv')
            plays['is_pass_play'] = plays['passResult'].notna().astype(int)
            return plays
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            sys.exit(1)

    def engineer_features(self, plays):
        logging.info("Engineering features...")
        features = plays[['gameId', 'playId', 'down', 'yardsToGo', 'yardlineNumber', 'quarter', 'gameClock', 'offenseFormation']]
        
        # Convert gameClock to seconds
        features['seconds_left'] = features['gameClock'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
        features = features.drop('gameClock', axis=1)
        
        # One-hot encode offenseFormation
        features = pd.get_dummies(features, columns=['offenseFormation'], prefix='formation')
        
        # Ensure all columns are numeric
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.to_numeric(features[col], errors='coerce')
        
        # Fill NaN values with median
        imputer = SimpleImputer(strategy='median')
        features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        
        self.feature_names = features.columns.tolist()
        self.X_data = features
        self.y_data = plays['is_pass_play']
        
        return features, plays['is_pass_play']

    def train_model(self):
        logging.info("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
        X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.selector.transform(X_test_scaled)

        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train_selected, y_train)

        y_pred = self.model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        logging.info(f"Model accuracy: {accuracy:.2f}")
        logging.info(f"ROC AUC: {roc_auc:.2f}")
        logging.info("\n" + classification_report(y_test, y_pred))

    def predict_play(self, features):
        for feature in self.feature_names:
            if feature not in features.columns:
                features[feature] = 0

        features = features[self.feature_names]

        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        prediction = self.model.predict(features_selected)[0]
        probability = self.model.predict_proba(features_selected)[0]
        return "Pass" if prediction == 1 else "Run", max(probability)

    def update_model(self, features, actual_outcome):
        new_X = pd.DataFrame([features], columns=self.feature_names)
        new_y = pd.Series([actual_outcome])
        self.X_data = pd.concat([self.X_data, new_X], ignore_index=True)
        self.y_data = pd.concat([self.y_data, new_y], ignore_index=True)
        self.train_model()
        logging.info("Model updated with new data and retrained.")

    def save_model(self, filename):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_names': self.feature_names,
            'X_data': self.X_data,
            'y_data': self.y_data
        }
        joblib.dump(model_data, filename)
        logging.info(f"Model saved to {filename}")

    def load_model(self, filename):
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.selector = model_data['selector']
            self.feature_names = model_data['feature_names']
            self.X_data = model_data['X_data']
            self.y_data = model_data['y_data']
            logging.info(f"Model loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading model from {filename}: {str(e)}")
            raise

def get_user_input(prompt, convert_func, valid_values=None):
    while True:
        value = input(prompt)
        if value.lower() in ['q', 'quit']:
            print("Quitting the program.")
            sys.exit(0)
        if value.lower() == 's':
            return None
        if value.lower() == 'c':
            print("Clearing all inputs. Starting over.")
            return 'clear'
        try:
            converted = convert_func(value)
            if valid_values is None or converted in valid_values:
                return converted
            else:
                print(f"Invalid input. Please enter one of these values: {valid_values}")
        except ValueError:
            print("Invalid input. Please try again.")

def get_user_input_realtime():
    inputs = {}
    prompts = [
        ("Down (1-4): ", 'down', lambda x: int(x), [1, 2, 3, 4]),
        ("Yards to go: ", 'yardsToGo', lambda x: float(x), None),
        ("Yard line: ", 'yardlineNumber', lambda x: float(x), None),
        ("Quarter (1-4): ", 'quarter', lambda x: int(x), [1, 2, 3, 4]),
        ("Time left (MM:SS): ", 'seconds_left', lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]), None),
        ("Formation (S/U/P/N/H): ", 'formation', lambda x: x.upper(), ['S', 'U', 'P', 'N', 'H'])
    ]

    for prompt_info in prompts:
        prompt, key, convert = prompt_info[:3]
        valid_values = prompt_info[3] if len(prompt_info) > 3 else None
        
        while True:
            value = get_user_input(prompt, convert, valid_values)
            if value == 'clear':
                return get_user_input_realtime()
            if value is not None:
                inputs[key] = value
            break

    return inputs

def get_user_input_advanced():
    inputs = {}
    coach_friendly_names = {
        'down': 'Down (1-4)',
        'yardsToGo': 'Yards to go',
        'yardlineNumber': 'Yard line',
        'quarter': 'Quarter (1-4)',
        'seconds_left': 'Seconds left in quarter',
        'formation_SHOTGUN': 'Is formation Shotgun? (y/n)',
        'formation_UNDER CENTER': 'Is formation Under Center? (y/n)',
        'formation_PISTOL': 'Is formation Pistol? (y/n)',
        'formation_NO HUDDLE SHOTGUN': 'Is formation No Huddle Shotgun? (y/n)',
        'formation_NO HUDDLE': 'Is formation No Huddle? (y/n)',
    }

    print("\nEnter detailed play information (or 's' to skip, 'c' to clear all inputs, 'q' to quit):")
    for feature in predictor.feature_names:
        if feature in coach_friendly_names:
            while True:
                if 'formation' in feature:
                    value = get_user_input(f"{coach_friendly_names[feature]}: ", lambda x: x.lower(), ['y', 'n'])
                else:
                    value = get_user_input(f"{coach_friendly_names[feature]}: ", float)
                
                if value == 'clear':
                    return get_user_input_advanced()
                if value is not None:
                    inputs[feature] = 1 if value == 'y' else 0 if 'formation' in feature else value
                break

    return inputs

def real_time_prediction():
    while True:
        print("\nEnter play details (or 's' to skip a field, 'c' to clear all inputs, 'q' to quit):")
        features = get_user_input_realtime()
        logging.debug(f"Raw input features: {features}")

        if 'formation' in features:
            formation = predictor.formation_map.get(features['formation'], '')
            for f in predictor.formation_map.values():
                features[f'formation_{f}'] = int(formation == f)
            features.pop('formation')
        logging.debug(f"Features after formation mapping: {features}")

        for feature in predictor.feature_names:
            if feature not in features:
                features[feature] = 0
        logging.debug(f"Final features before prediction: {features}")

        try:
            prediction, confidence = predictor.predict_play(pd.DataFrame([features]))
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence:.2f}")
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            print("An error occurred during prediction. Please check the log for details.")

        choice = input("Press Enter to predict another play, or type 'q' to quit: ")
        if choice.lower() == 'q':
            break

def advanced_mode():
    while True:
        features = get_user_input_advanced()
        
        prediction, confidence = predictor.predict_play(pd.DataFrame([features]))
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        
        actual = get_user_input("Was it actually a pass play? (y/n, or 's' to skip): ", lambda x: x.lower(), ['y', 'n', 's'])
        if actual == 'y':
            predictor.update_model(features, 1)
        elif actual == 'n':
            predictor.update_model(features, 0)
        
        choice = input("Press Enter to continue, or type 'q' to quit: ")
        if choice.lower() == 'q':
            break

if __name__ == "__main__":
    predictor = NFLPlayPredictor()
    try:
        predictor.load_model('nfl_predictor_model.joblib')
        print("Loaded pre-trained model.")
    except Exception as e:
        print(f"Error loading pre-trained model: {str(e)}")
        print("Training a new model...")
        plays = predictor.load_data()
        predictor.engineer_features(plays)
        predictor.train_model()
        predictor.save_model('nfl_predictor_model.joblib')
        print("New model trained and saved.")

    while True:
        mode = input("Select mode (1 for quick prediction, 2 for detailed analysis, q to quit): ")
        if mode == '1':
            real_time_prediction()
        elif mode == '2':
            advanced_mode()
        elif mode.lower() == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid mode selected. Please try again.")