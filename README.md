# Airline Data Clustering Project

## Objective
Perform clustering (K-means clustering and DBSCAN) for the airline data to obtain the optimum number of clusters. Draw inferences from the clusters obtained.

## Data Description
The dataset contains information on flight details, including departure and arrival times, delays, and other relevant metrics. The goal is to identify clusters of flights with similar characteristics for analysis.

### Columns
- **Year**: Year of the flight
- **Month**: Month of the flight
- **DayofMonth**: Day of the month of the flight
- **DayOfWeek**: Day of the week of the flight
- **DepTime**: Departure time (local time)
- **ArrTime**: Arrival time (local time)
- **UniqueCarrier**: Unique carrier code
- **FlightNum**: Flight number
- **TailNum**: Tail number of the aircraft
- **ActualElapsedTime**: Actual elapsed time of the flight (in minutes)
- **AirTime**: Time spent in the air (in minutes)
- **ArrDelay**: Arrival delay (in minutes)
- **DepDelay**: Departure delay (in minutes)
- **Origin**: Origin airport code
- **Dest**: Destination airport code
- **Distance**: Distance between origin and destination (in miles)
- **TaxiIn**: Taxi-in time (in minutes)
- **TaxiOut**: Taxi-out time (in minutes)
- **Cancelled**: Flight cancellation indicator (1 if cancelled, 0 otherwise)
- **CancellationCode**: Reason for cancellation (A: carrier, B: weather, C: NAS, D: security)
- **Diverted**: Flight diversion indicator (1 if diverted, 0 otherwise)

## Steps to Perform Clustering
1. **Data Preprocessing**: Clean the data by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Determine Optimal Clusters**: Use methods like the Elbow Method or Silhouette Score to determine the optimal number of clusters.
3. **Apply K-means Clustering**: Perform K-means clustering on the preprocessed data.
4. **Analyze Clusters**: Draw inferences from the clusters obtained and interpret the results.

## Inferences
- **Cluster Characteristics**: Describe the characteristics of each cluster, such as average delays, common routes, etc.
- **Insights**: Provide insights based on the clustering results, such as identifying patterns in flight delays or common factors in cancellations.

## Usage
1. **Clone Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/muzammiltariq95/clustering-analysis.git
   ```
2. **Install Dependencies**: Install the required dependencies using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Analysis**: Execute the provided scripts to perform clustering and analyze the results.

## Contributing
Contributions are welcome! Please read the contributing guidelines for more details.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
