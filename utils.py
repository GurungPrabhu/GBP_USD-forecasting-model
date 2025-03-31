from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def split_train_validate_test(df: DataFrame, train_percent=.6, validate_percent=.2, test_percent=.2):
    assert train_percent + validate_percent + test_percent == 1, "The sum of train, validate, and test percentages must be 1."
    
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    
    train = df.iloc[:train_end]
    validate = df.iloc[train_end:validate_end]
    test = df.iloc[validate_end:]
    
    return train, validate, test

class DataPreprocessor:
    def __init__(self, target='Open'):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target = target

    def fit(self, train_data: DataFrame):
        self.scaler.fit(train_data[['Open', 'Time']])

    def transform(self, data: DataFrame):
        y = data[self.target].values
        X = self.scaler.transform(data[['Open', 'Time']])
        X = X.reshape(-1, 2)
        return X, y

    def inverse_transform(self, X):
        X = X.reshape(-1, 2)
        return self.scaler.inverse_transform(X)



def plot_history(history):
    history_dict = history.history

    #MAE
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 1)
    plt.plot(history_dict['mae'], label='Training MAE')
    plt.plot(history_dict['val_mae'], label='Validation MAE')

    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')

    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)

    #MSE
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2)
    plt.plot(history_dict['mse'], label='Training MSE')
    plt.plot(history_dict['val_mse'], label='Training MSE')

    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training and Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

