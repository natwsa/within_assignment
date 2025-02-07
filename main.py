import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Define manually constructed PATH station data with coordinates and connections
stations = {
    'Newark': (-74.1724, 40.7357, ['NJ Transit', 'Amtrak', 'Bus'], ['Harrison'], ['Red', 'Orange']),
    'Harrison': (-74.1557, 40.7465, ['NJ Transit', 'Bus'], ['Newark', 'Journal Square'], ['Red', 'Orange']),
    'Journal Square': (-74.0631, 40.7322, ['Bus'], ['Harrison', 'Grove Street'], ['Red', 'Orange']),
    'Grove Street': (-74.0413, 40.7196, ['Bus'], ['Journal Square', 'Exchange Place'], ['Red', 'Green']),
    'Exchange Place': (-74.0327, 40.7166, ['Ferry', 'Bus'], ['Grove Street', 'World Trade Center', 'Newport'], ['Red', 'Green']),
    'Newport': (-74.0384, 40.7269, ['Light Rail', 'Bus'], ['Hoboken', 'Exchange Place', 'Grove Street'], ['Green', 'Orange']),
    'Hoboken': (-74.0321, 40.7359, ['NJ Transit', 'Amtrak', 'Ferry', 'Bus'], ['Newport', '33rd Street'], ['Green', 'Orange']),
    'World Trade Center': (-74.0121, 40.7115, ['NYC Subway', 'Bus'], ['Exchange Place'], ['Red']),
    'Christopher Street': (-74.0048, 40.7328, ['NYC Subway', 'Bus'], ['9th Street'], ['Green']),
    '9th Street': (-73.9981, 40.7349, ['NYC Subway', 'Bus'], ['Christopher Street', '14th Street'], ['Green']),
    '14th Street': (-73.9962, 40.7379, ['NYC Subway', 'Bus'], ['9th Street', '23rd Street'], ['Green']),
    '23rd Street': (-73.9950, 40.7428, ['NYC Subway', 'Bus'], ['14th Street', '33rd Street'], ['Green']),
    '33rd Street': (-73.9879, 40.7488, ['NYC Subway', 'Bus'], ['23rd Street', 'Hoboken'], ['Green', 'Orange'])
}



# Daily riders data
daily_riders = {
    'Christopher Street': 3147,
    '9th Street': 2792,
    '14th Street': 4834,
    '23rd Street': 5033,
    '33rd Street': 19543,
    'Newark': 14267,
    'Harrison': 5417,
    'Journal Square': 18174,
    'Grove Street': 14545,
    'Exchange Place': 10112,
    'Newport': 10678,
    'Hoboken': 15943,
    'World Trade Center': 37278
}

# Convert station dictionary into DataFrame
stations_df = pd.DataFrame(
    [(key, value[0], value[1], len(value[2]), len(value[3]), value[4], daily_riders.get(key, None)) for key, value in stations.items()],
    columns=['STATION', 'Longitude', 'Latitude', 'TRANSIT_CONS', 'CONNECTIONS', 'LINES', 'DAILY_RIDERS']
)

stations_df['IS_NY'] = stations_df['STATION'].apply(lambda x: 1 if x in ['World Trade Center', 'Christopher Street', '9th Street', '14th Street', '23rd Street', '33rd Street'] else 0)

# Compute distance to the nearest station
def calculate_nearest_distance(latitude, longitude, stations_df):
    distances = [
        geodesic((latitude, longitude), (row['Latitude'], row['Longitude'])).miles
        for _, row in stations_df.iterrows() if (row['Latitude'], row['Longitude']) != (latitude, longitude)
    ]
    return min(distances) if distances else None

stations_df['PROX'] = stations_df.apply(lambda row: calculate_nearest_distance(
    row['Latitude'], row['Longitude'], stations_df), axis=1)

# Count the number of lines each station serves
stations_df['LINES_COUNT'] = stations_df['LINES'].apply(len)

# Keep only relevant columns
final_df = stations_df[['STATION', 'TRANSIT_CONS', 'CONNECTIONS', 'PROX', 'LINES_COUNT', 'DAILY_RIDERS', 'IS_NY']]

# Prepare data for machine learning
X = final_df[['TRANSIT_CONS', 'CONNECTIONS', 'PROX', 'LINES_COUNT', 'IS_NY']]

y = final_df['DAILY_RIDERS']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=50, random_state=10)
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
dummy_mae = mean_absolute_error(y_test, [10000, 12000, 14000])

# Print model evaluation
print("Mean Absolute Error:", mae)

# Compute feature importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()

# Print model evaluation
print("Mean Absolute Error:", mae)

# Display the final dataframe
print("Final Merged DataFrame:")
print(final_df.head())
