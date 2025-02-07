import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Convert station dictionary into DataFrame
stations_df = pd.DataFrame(
    [(key, value[0], value[1], len(value[2]), len(value[3]), value[4], daily_riders.get(key, None)) for key, value in stations.items()],
    columns=['STATION', 'Longitude', 'Latitude', 'TRANSIT_CONS', 'CONNECTIONS', 'LINES', 'DAILY_RIDERS']
)

# Add feature variable
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

# Keep only potentially relevant feature columns
final_df = stations_df[['STATION', 'TRANSIT_CONS', 'CONNECTIONS', 'PROX', 'LINES_COUNT', 'DAILY_RIDERS', 'IS_NY']]

# Split features from target
X = final_df[['TRANSIT_CONS', 'CONNECTIONS', 'PROX', 'LINES_COUNT', 'IS_NY']]

y = final_df['DAILY_RIDERS']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#Train random forest models with different numbers of n_estimators
mae_values = []
n_estimators_range = range(1, 201)
for n in n_estimators_range:
    rf_model = RandomForestRegressor(n_estimators=n, random_state=12)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mae_values.append(mean_absolute_error(y_test, y_pred))

# Plot MAE vs n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, mae_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Absolute Error')
plt.title('MAE as a Function of n_estimators')
plt.grid(True)
plt.show()

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
