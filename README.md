# Chicago Crime Clustering Analysis

## 🧭 Project Motivation
As a student focused on **Data Science and Public Safety**, I undertook this project with a vision: **Can crime data tell us where and when to make our cities safer?**

By leveraging clustering algorithms on real-world crime data from Chicago, I aimed to:
- Detect crime **patterns** based on time, location, and type.
- Recommend **targeted interventions** for each pattern.
- Build a foundation for **policy, enforcement, and community awareness**.

> "A safer society begins with understanding its risks." 

---

## 📂 Dataset Overview
- **Source**: [Kaggle - Chicago Crime Dataset 2001-present](https://www.kaggle.com/datasets/nathaniellybrand/chicago-crime-dataset-2001-present)
- **Size**: 7.8 million+ records
- **Fields Used**: `Date`, `Primary Type`, `Location Description`, `Arrest`, `Domestic`, `Year`

---

## 🔍 Exploratory Data Analysis (EDA)

### 📌 Dataset Summary
```python
chicago_df = pd.read_csv(csv_path)
print(chicago_df.shape)  # (7846809, 22)
chicago_df.info()
```
- 22 columns, 7.8M records.
- Mixed types: object, int, float, bool.

### 📌 Statistical Overview
```python
chicago_df.describe()
```
- Fields like `ID`, `Beat`, `District`, and coordinates show normal ranges.
- Some missing values in location-related features.

### 📌 Correlation Matrix
```python
import plotly.express as px
corr = chicago_df.corr(numeric_only=True)
print(corr)
```
- Strong correlation between `Beat` and `District`.
- Minor correlation between `Domestic` and `Community Area`.

---

## 🔧 Step-by-Step Preprocessing

### 🔹 1. Column Selection
We drop unnecessary or identifying columns:
```python
chicago_df = chicago_df[['Date', 'Primary Type', 'Location Description', 'Arrest', 'Domestic', 'Year']]
```

### 🔹 2. Missing Value Handling
```python
print(chicago_df.shape)
print(chicago_df.isnull().sum())# Location Description = 10758
print(chicago_df.value_counts())
chicago_df['Location Description'] = chicago_df['Location Description'].fillna('OTHER')
```

### 🔹 3. Location Grouping
We reduced 100+ location strings into 10 groups using `categorize_location()`.
```python
def categorize_location(desc):
    desc = str(desc).lower()

    if any(x in desc for x in ['residence', 'apartment', 'yard', 'porch', 'garage', 'vestibule', 'coach house', 'rooming house', 'cha hallway', 'cha apartment', 'cha play', 'cha parking', 'cha lobby', 'cha stairwell', 'cha elevator', 'cha breezeway', 'cha grounds']):
        return 'RESIDENTIAL'
    elif any(x in desc for x in ['store', 'shop', 'retail', 'restaurant', 'tavern', 'bar', 'motel', 'hotel', 'liquor store', 'gas station', 'atm', 'bank', 'funeral', 'laundry', 'cleaning', 'dealership', 'currency exchange', 'beauty salon', 'barber', 'appliance']):
        return 'COMMERCIAL'
    elif any(x in desc for x in ['cta', 'station', 'platform', 'train', 'bus', 'taxi', 'vehicle', 'garage', 'parking lot', 'airport', 'delivery truck', 'ride share', 'expressway', 'tracks', 'highway', 'uber', 'lyft', 'transportation system', 'trolley']):
        return 'TRANSPORT'
    elif any(x in desc for x in ['street', 'sidewalk', 'park', 'property', 'alley', 'bridge', 'river', 'lake', 'forest', 'beach', 'lagoon', 'riverbank', 'lakefront', 'wooded', 'gangway', 'sewer', 'prairie']):
        return 'PUBLIC'
    elif any(x in desc for x in ['school', 'college', 'university', 'grammar school', 'high school', 'school yard', 'day care']):
        return 'EDUCATION'
    elif any(x in desc for x in ['hospital', 'medical', 'dental', 'nursing', 'retirement', 'ymca', 'animal hospital', 'funeral']):
        return 'MEDICAL'
    elif any(x in desc for x in ['police', 'jail', 'lock-up', 'courthouse', 'government', 'fire station', 'federal', 'county']):
        return 'GOVERNMENT'
    elif any(x in desc for x in ['warehouse', 'factory', 'manufacturing', 'construction', 'trucking', 'appliance', 'cleaners', 'garage/auto', 'junk yard', 'loading dock']):
        return 'INDUSTRIAL'
    elif any(x in desc for x in ['club', 'athletic', 'pool', 'sports arena', 'bowling', 'movie', 'theater', 'lounge', 'banquet', 'gym']):
        return 'RECREATIONAL'
    else:
        return 'OTHER'

chicago_df['Location Group'] = chicago_df['Location Description'].apply(categorize_location)
chicago_df['Location Group'].value_counts()
chicago_df.drop(columns = ["Location Description"], inplace = True)
```

### 🔹 4. Crime Type Grouping
Simplified 30+ crime categories into fewer major types.
```python
def categorize_crime_type(crime):
    crime = crime.strip().upper()

    if crime in ['BATTERY', 'ASSAULT', 'HOMICIDE', 'ROBBERY', 'KIDNAPPING', 'ARSON', 'INTIMIDATION', 'STALKING', 'DOMESTIC VIOLENCE']:
        return 'VIOLENT'
    elif crime in ['THEFT', 'BURGLARY', 'MOTOR VEHICLE THEFT', 'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS']:
        return 'PROPERTY'
    elif crime in ['NARCOTICS', 'OTHER NARCOTIC VIOLATION']:
        return 'DRUGS'
    elif crime in ['CRIM SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT', 'SEX OFFENSE', 'OBSCENITY', 'PUBLIC INDECENCY']:
        return 'SEXUAL'
    elif crime in ['DECEPTIVE PRACTICE', 'CONCEALED CARRY LICENSE VIOLATION', 'LIQUOR LAW VIOLATION', 'GAMBLING', 'PROSTITUTION']:
        return 'FRAUD'
    elif crime in ['PUBLIC PEACE VIOLATION', 'WEAPONS VIOLATION', 'INTERFERENCE WITH PUBLIC OFFICER', 'HUMAN TRAFFICKING', 'RITUALISM']:
        return 'PUBLIC ORDER'
    elif crime in ['OFFENSE INVOLVING CHILDREN']:
        return 'CHILD/FAMILY'
    elif 'NON-CRIMINAL' in crime:
        return 'NON-CRIMINAL'
    else:
        return 'OTHER'

chicago_df["Crime Group"] = chicago_df["Primary Type"].apply(categorize_crime_type)
chicago_df["Crime Group"].value_counts()
```

### 🔹 5. Date → Month
```python
chicago_df['Date'] = pd.to_datetime(chicago_df['Date'], format='%m/%d/%Y %I:%M:%S %p')
chicago_df['Month'] = chicago_df['Date'].dt.month
chicago_df.drop(columns=['Date', 'Year'], inplace=True)
```

### 🔹 6. Convert Booleans and Plot Histograms
```python
chicago_df['Arrest'] = chicago_df['Arrest'].astype(int)
chicago_df['Domestic'] = chicago_df['Domestic'].astype(int)

for col in chicago_df.select_dtypes(include='number'):
    chicago_df[col].plot(kind='hist', bins=30, title=col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
```

### 🔹 7. One-Hot Encoding
```python
chicago_df_en = pd.get_dummies(chicago_df, columns=['Location Group', 'Crime Group', 'Month'])
```

### 🔹 8. Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
chicago_scaled = pd.DataFrame(scaler.fit_transform(chicago_df_en), columns=chicago_df_en.columns)
```

### 🔹 9. Scatter
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2) # For Graph
reduced = pca.fit_transform(chicago_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.5)
plt.title('Chicago Crime Data (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/a897e727-5c33-4613-ba69-071e2d6ec755)
📌 Although PCA is not optimal for clustering, we applied it for visualization

---

## 🔵 K-Means Clustering

### 🔍 Finding Optimal K using Silhouette Score
```python
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sample_data = resample(chicago_scaled, n_samples=30000, random_state=0)

for k in range(20, 30):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(sample_data)
    score = silhouette_score(sample_data, labels)
    print(f'k={k}, silhouette : {score:.4f}')
```
> 📌 Best score: `k=24`

### 🔧 Final KMeans Clustering
```python
kmeans = KMeans(n_clusters=24, init='k-means++', max_iter=300, random_state=0)
kmeans_labels = kmeans.fit_predict(sample_data)
```

### 📌 Cluster Summary + Visualization
We find top `Location`, `Crime`, `Month` for each cluster:
```python
clustered_df = pd.DataFrame(sample_data, columns=chicago_scaled.columns)
clustered_df['cluster'] = kmeans_labels

location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if 'Crime Group_' in col]
month_col = [col for col in clustered_df.columns if 'Month_' in col]

cluster_summary = pd.DataFrame({
    "Top Location": clustered_df.groupby('cluster')[location_col].mean().idxmax(axis=1),
    "Top_Crime": clustered_df.groupby('cluster')[crime_col].mean().idxmax(axis=1),
    "Top_Month": clustered_df.groupby('cluster')[month_col].mean().idxmax(axis=1)
})
print(cluster_summary)
```

## 🎯 3. K-Means Clustering

### 3.1 Optimal Cluster Count (Silhouette Score)
```python
sample_data = resample(chicago_scaled, n_samples=30000, random_state=0)

for k in range(20, 30):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(sample_data)
    score = silhouette_score(sample_data, labels)
    print(f'k={k}, silhouette : {score:.4f}')
```
📌 Best k: **24**, Silhouette: 0.2944

### 3.2 Final Clustering
```python
kmeans = KMeans(n_clusters=24, random_state=0)
kmeans_labels = kmeans.fit_predict(sample_data)
```

### 3.3 Cluster Profiling
```python
clustered_df = pd.DataFrame(sample_data, columns=chicago_scaled.columns)
clustered_df['cluster'] = kmeans_labels

location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if 'Crime Group_' in col]
month_col = [col for col in clustered_df.columns if 'Month_' in col]

location_summary = clustered_df.groupby('cluster')[location_col].mean()
crime_summary = clustered_df.groupby('cluster')[crime_col].mean()
month_summary = clustered_df.groupby('cluster')[month_col].mean()

cluster_summary = pd.DataFrame({
    "Top Location": location_summary.idxmax(axis=1),
    "Top_Crime": crime_summary.idxmax(axis=1),
    "Top_Month": month_summary.idxmax(axis=1)
})
print(cluster_summary)
```

### 3.4 Strategic Insight (Examples)
```
                        Top Location                 Top_Crime Top_Month
cluster                                                                 
0          Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_4
1          Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_9
2          Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_8
3         Location Group_RESIDENTIAL      Crime Group_PROPERTY   Month_6
4          Location Group_COMMERCIAL      Crime Group_PROPERTY  Month_11
5          Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_5
6          Location Group_COMMERCIAL         Crime Group_DRUGS   Month_2
7          Location Group_COMMERCIAL      Crime Group_PROPERTY  Month_10
8          Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_1
9          Location Group_GOVERNMENT         Crime Group_OTHER   Month_2
10         Location Group_INDUSTRIAL      Crime Group_PROPERTY   Month_4
11        Location Group_RESIDENTIAL       Crime Group_VIOLENT   Month_5
12             Location Group_PUBLIC  Crime Group_PUBLIC ORDER   Month_5
13        Location Group_RESIDENTIAL  Crime Group_CHILD/FAMILY   Month_1
14         Location Group_COMMERCIAL      Crime Group_PROPERTY   Month_3
15             Location Group_PUBLIC         Crime Group_DRUGS   Month_3
16        Location Group_RESIDENTIAL         Crime Group_OTHER   Month_2
17          Location Group_EDUCATION       Crime Group_VIOLENT   Month_3
18            Location Group_MEDICAL       Crime Group_VIOLENT  Month_11
19        Location Group_RESIDENTIAL        Crime Group_SEXUAL   Month_1
20        Location Group_RESIDENTIAL       Crime Group_VIOLENT  Month_12
21        Location Group_RESIDENTIAL       Crime Group_VIOLENT   Month_4
22             Location Group_PUBLIC       Crime Group_VIOLENT   Month_7
23       Location Group_RECREATIONAL      Crime Group_PROPERTY   Month_8
```
- Cluster 3: `RESIDENTIAL`, `PROPERTY`, `Month_6` → Increase patrols in residential areas during June.
- Cluster 11: `RESIDENTIAL`, `VIOLENT`, `Month_5` → Domestic violence awareness month.
- Cluster 19: `RESIDENTIAL`, `SEXUAL`, `Month_1` → Wintertime victim support services.

### 3.5 PCA Visualization
```python
pca = PCA(n_components=2)
reduced = pca.fit_transform(clustered_df.drop('cluster', axis=1))

plt.figure(figsize=(10,6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clustered_df['cluster'], cmap='tab10', s=10)
plt.title('KMeans Clustering (PCA Reduced)')
plt.colorbar()
plt.show()
```
![image](https://github.com/user-attachments/assets/ee24ca2a-aa49-4638-b29b-de2d5463cd0a)

---

## 🔁 4. Mean Shift Clustering
```python
sample_data = resample(chicago_scaled, n_samples=10000, random_state=42)
bandwidth = estimate_bandwidth(sample_data, quantile=0.1)
meanshift = MeanShift(bandwidth=6)
labels = meanshift.fit_predict(sample_data)
```
📌 Detected: 16 clusters

### Strategic Examples:
```
                      Top Location                 Top_Crime Top_Month
cluster                                                                 
0          Location Group_COMMERCIAL         Crime Group_FRAUD   Month_6
1           Location Group_EDUCATION       Crime Group_VIOLENT   Month_2
2              Location Group_PUBLIC  Crime Group_PUBLIC ORDER   Month_8
3         Location Group_RESIDENTIAL        Crime Group_SEXUAL   Month_1
4             Location Group_MEDICAL       Crime Group_VIOLENT   Month_8
5          Location Group_INDUSTRIAL      Crime Group_PROPERTY   Month_9
6         Location Group_RESIDENTIAL  Crime Group_CHILD/FAMILY  Month_12
7        Location Group_RECREATIONAL      Crime Group_PROPERTY   Month_3
8           Location Group_EDUCATION  Crime Group_PUBLIC ORDER  Month_10
9           Location Group_EDUCATION        Crime Group_SEXUAL  Month_12
10            Location Group_MEDICAL        Crime Group_SEXUAL   Month_9
11         Location Group_GOVERNMENT  Crime Group_PUBLIC ORDER  Month_10
12         Location Group_GOVERNMENT       Crime Group_VIOLENT   Month_7
13       Location Group_RECREATIONAL  Crime Group_PUBLIC ORDER   Month_4
14       Location Group_RECREATIONAL        Crime Group_SEXUAL   Month_4
15          Location Group_EDUCATION  Crime Group_CHILD/FAMILY  Month_10
```
- Cluster 0: `COMMERCIAL`, `FRAUD`, `Month_6` → Audit programs for summer sales fraud.
- Cluster 4: `MEDICAL`, `VIOLENT`, `Month_8` → Strengthen hospital security protocols.
  
![image](https://github.com/user-attachments/assets/7f84f2ba-fd1a-46cc-8da1-5de565742e51)

---

## 🔀 5. Gaussian Mixture Model (GMM)
```python
gmm = GaussianMixture(n_components=8, random_state=0)
gmm_labels = gmm.fit_predict(sample_data)
```
### Examples:
```
                      Top Location             Top_Crime Top_Month
cluster                                                            
0         Location Group_COMMERCIAL     Crime Group_FRAUD   Month_1
1             Location Group_PUBLIC     Crime Group_DRUGS   Month_5
2            Location Group_MEDICAL     Crime Group_OTHER   Month_5
3        Location Group_RESIDENTIAL   Crime Group_VIOLENT  Month_10
4         Location Group_INDUSTRIAL  Crime Group_PROPERTY  Month_12
5          Location Group_EDUCATION     Crime Group_OTHER   Month_4
6         Location Group_COMMERCIAL  Crime Group_PROPERTY   Month_6
7             Location Group_PUBLIC     Crime Group_DRUGS  Month_11
```
- Cluster 0: `COMMERCIAL`, `FRAUD`, `Month_1` → Holiday scams in January.
- Cluster 3: `RESIDENTIAL`, `VIOLENT`, `Month_10` → Night patrols in October.

![image](https://github.com/user-attachments/assets/14d197ce-d6a8-41a9-bc68-f29cc6c6c68e)
---

## 📉 6. DBSCAN Clustering
```python
dbscan = DBSCAN(eps=0.6, min_samples=8)
db_labels = dbscan.fit_predict(sample_data)
```
### Example Strategies:
```
                        Top Location                 Top_Crime Top_Month
cluster                                                                 
-1         Location Group_GOVERNMENT        Crime Group_SEXUAL  Month_10
 0         Location Group_COMMERCIAL         Crime Group_FRAUD   Month_6
 1             Location Group_PUBLIC  Crime Group_PUBLIC ORDER   Month_8
 2        Location Group_RESIDENTIAL        Crime Group_SEXUAL   Month_1
 3            Location Group_MEDICAL       Crime Group_VIOLENT   Month_8
 4        Location Group_RESIDENTIAL  Crime Group_CHILD/FAMILY  Month_12
 5          Location Group_EDUCATION       Crime Group_VIOLENT   Month_2
 6         Location Group_INDUSTRIAL      Crime Group_PROPERTY   Month_9
 7       Location Group_RECREATIONAL      Crime Group_PROPERTY   Month_3
 8          Location Group_EDUCATION  Crime Group_PUBLIC ORDER  Month_10
```
- Cluster 1: `PUBLIC`, `PUBLIC ORDER`, `Month_8` → Public disorder campaigns.
- Cluster 2: `RESIDENTIAL`, `SEXUAL`, `Month_1` → Victim advocacy early in year.
  
![image](https://github.com/user-attachments/assets/6b15d1d9-7c38-44f0-861e-ac6d10a79c45)

---

## 📌 Conclusion & Impact
- ✔️ Processed 7.8M records from raw CSV to clean ML input.
- ✔️ Tried **KMeans**, **MeanShift**, **GMM**, and **DBSCAN**.
- ✔️ Cluster insights mapped to real-world interventions.

### 🚨 Potential Impact
- Inform **city policing schedules**.
- Support **AI crime monitoring** tools.
- Help design **proactive safety campaigns**.
