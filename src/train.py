import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import TensorBoard, EarlyStopping
from io_utils import save_pickle
from config import DATA_PATH, LOG_DIR, MODEL_PATH, LABEL_ENCODER_FILEPATH, ONE_HOT_ENCODER_FILEPATH, STANDARD_SCALER_FILEPATH, HISTOGRAM_FREQUENCY, MONITOR, PATIENCE, HIDDEN_LAYER_ONE_NEURONS, HIDDEN_LAYER_TWO_NEURONS, OUTPUT_LAYER_NEURONS, HIDDEN_LAYER_ACTIVATION, OUTPUT_LAYER_ACTIVATION, LEARNING_RATE, METRICS, EPOCHS

class Trainer:
    def __init__(self):
        self.dataset = pd.read_csv(DATA_PATH)
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()
        self.standard_scaler = StandardScaler()
        self.tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=HISTOGRAM_FREQUENCY)
        self.early_stopping_callback = EarlyStopping(monitor=MONITOR, patience=PATIENCE, restore_best_weights=True)
    
    def run(self):
        self.preprocessing()
        self.train_test_split()
        self.scaler()
        self.build_and_compile_model()
        self.train_model()
        self.save_model()

    def preprocessing(self):
        self.dataset = self.dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        self.dataset['Gender'] = self.label_encoder.fit_transform(self.dataset['Gender'])
        save_pickle(LABEL_ENCODER_FILEPATH, self.label_encoder)

        encoded_geo_df = pd.DataFrame(self.one_hot_encoder.fit_transform(self.dataset[['Geography']]).toarray(), columns=self.one_hot_encoder.get_feature_names_out(['Geography']))
        self.dataset = pd.concat([self.dataset.drop('Geography', axis=1), encoded_geo_df], axis=1)
        save_pickle(ONE_HOT_ENCODER_FILEPATH, self.one_hot_encoder)

    def train_test_split(self):
        X = self.dataset.drop('Exited', axis=1)
        y = self.dataset['Exited']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def scaler(self):
        self.X_train = self.standard_scaler.fit_transform(self.X_train)
        self.X_test = self.standard_scaler.transform(self.X_test)
        save_pickle(STANDARD_SCALER_FILEPATH, self.standard_scaler)

    def build_and_compile_model(self):
        self.model = Sequential([
            Input(shape=(self.X_train.shape[1],)),
            Dense(HIDDEN_LAYER_ONE_NEURONS, activation=HIDDEN_LAYER_ACTIVATION),
            Dense(HIDDEN_LAYER_TWO_NEURONS, activation=HIDDEN_LAYER_ACTIVATION),
            Dense(OUTPUT_LAYER_NEURONS, activation=OUTPUT_LAYER_ACTIVATION)
        ])

        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.loss = BinaryCrossentropy()

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=METRICS)

    def train_model(self):
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=EPOCHS, callbacks=[self.tensorboard_callback, self.early_stopping_callback])
        print(history)

    def save_model(self):
        self.model.save(MODEL_PATH)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
