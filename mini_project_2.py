from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
# 실루엣 분석 metric 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_path = '/root/.cache/kagglehub/datasets/nathaniellybrand/chicago-crime-dataset-2001-present/versions/1/Crimes_-_2001_to_Present.csv'
chicago_df = pd.read_csv(csv_path)
print(chicago_df.shape)
# (7846809, 22)

## 1. EDA
### 1) Data check
chicago_df.head(3)
chicago_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7846809 entries, 0 to 7846808
Data columns (total 22 columns):
 #   Column                Dtype  
---  ------                -----  
 0   ID                    int64  
 1   Case Number           object 
 2   Date                  object 
 3   Block                 object 
 4   IUCR                  object 
 5   Primary Type          object 
 6   Description           object 
 7   Location Description  object 
 8   Arrest                bool   
 9   Domestic              bool   
 10  Beat                  int64  
 11  District              float64
 12  Ward                  float64
 13  Community Area        float64
 14  FBI Code              object 
 15  X Coordinate          float64
 16  Y Coordinate          float64
 17  Year                  int64  
 18  Updated On            object 
 19  Latitude              float64
 20  Longitude             float64
 21  Location              object 
dtypes: bool(2), float64(7), int64(3), object(10)
memory usage: 1.2+ GB
"""

chicago_df.describe()
"""

ID	Beat	District	Ward	Community Area	X Coordinate	Y Coordinate	Year	Latitude	Longitude
count	7.846809e+06	7.846809e+06	7.846762e+06	7.231960e+06	7.233333e+06	7.758698e+06	7.758698e+06	7.846809e+06	7.758698e+06	7.758698e+06
mean	7.074220e+06	1.185677e+03	1.129493e+01	2.275683e+01	3.747487e+01	1.164606e+06	1.885794e+06	2.010047e+03	4.184221e+01	-8.767147e+01
std	3.530056e+06	7.032093e+02	6.953759e+00	1.385113e+01	2.154184e+01	1.684276e+04	3.227139e+04	6.341715e+00	8.878488e-02	6.107057e-02
min	6.340000e+02	1.110000e+02	1.000000e+00	1.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	2.001000e+03	3.661945e+01	-9.168657e+01
25%	3.815790e+06	6.210000e+02	6.000000e+00	1.000000e+01	2.300000e+01	1.152985e+06	1.859077e+06	2.005000e+03	4.176873e+01	-8.771365e+01
50%	7.075819e+06	1.034000e+03	1.000000e+01	2.300000e+01	3.200000e+01	1.166121e+06	1.890736e+06	2.009000e+03	4.185593e+01	-8.766582e+01
75%	1.021223e+07	1.731000e+03	1.700000e+01	3.400000e+01	5.700000e+01	1.176384e+06	1.909282e+06	2.015000e+03	4.190680e+01	-8.762817e+01
max	1.314808e+07	2.535000e+03	3.100000e+01	5.000000e+01	7.700000e+01	1.205119e+06	1.951622e+06	2.023000e+03	4.202291e+01	-8.752453e+01
"""

import plotly.express as px

corr = chicago_df.corr(numeric_only=True)
print(corr)
"""
                      ID    Arrest  Domestic      Beat  District      Ward  \
ID              1.000000 -0.102899  0.063216 -0.038557 -0.004151  0.021980   
Arrest         -0.102899  1.000000 -0.066690 -0.014297 -0.016836 -0.016591   
Domestic        0.063216 -0.066690  1.000000 -0.045930 -0.042296 -0.056127   
Beat           -0.038557 -0.014297 -0.045930  1.000000  0.946903  0.642492   
District       -0.004151 -0.016836 -0.042296  0.946903  1.000000  0.685152   
Ward            0.021980 -0.016591 -0.056127  0.642492  0.685152  1.000000   
Community Area -0.014187 -0.001010  0.077876 -0.499337 -0.493973 -0.529376   
X Coordinate    0.007499 -0.032500  0.007815 -0.491908 -0.539796 -0.446450   
Y Coordinate    0.000153  0.000142 -0.082290  0.620989  0.628169  0.636963   
Year            0.984196 -0.107191  0.065460 -0.039768 -0.005285  0.021200   
Latitude        0.000118  0.000274 -0.082111  0.621366  0.628721  0.637100   
Longitude       0.007598 -0.032853  0.006604 -0.488491 -0.536866 -0.442303   

                Community Area  X Coordinate  Y Coordinate      Year  \
ID                   -0.014187      0.007499      0.000153  0.984196   
Arrest               -0.001010     -0.032500      0.000142 -0.107191   
Domestic              0.077876      0.007815     -0.082290  0.065460   
Beat                 -0.499337     -0.491908      0.620989 -0.039768   
District             -0.493973     -0.539796      0.628169 -0.005285   
Ward                 -0.529376     -0.446450      0.636963  0.021200   
Community Area        1.000000      0.248815     -0.754484 -0.014890   
X Coordinate          0.248815      1.000000     -0.460458  0.009322   
Y Coordinate         -0.754484     -0.460458      1.000000  0.000750   
Year                 -0.014890      0.009322      0.000750  1.000000   
Latitude             -0.753535     -0.462374      0.999994  0.000707   
Longitude             0.240546      0.999830     -0.455475  0.009446   

                Latitude  Longitude  
ID              0.000118   0.007598  
Arrest          0.000274  -0.032853  
Domestic       -0.082111   0.006604  
Beat            0.621366  -0.488491  
District        0.628721  -0.536866  
Ward            0.637100  -0.442303  
Community Area -0.753535   0.240546  
X Coordinate   -0.462374   0.999830  
Y Coordinate    0.999994  -0.455475  
Year            0.000707   0.009446  
Latitude        1.000000  -0.457445  
Longitude      -0.457445   1.000000 
"""

## 2. Data Preprocessing
### 1) Drop useless Data -> ID, CaseNumber, IUCR, Description, Ward, Community Area,
### FBI Code, X coordinate, Y coordinate, Updated On, Latitude, Longitude, Location
chicago_df = chicago_df[['Date', 'Primary Type', 'Location Description', 'Arrest', 'Domestic', 'Year']]
chicago_df.head(3)
"""
	Date	Primary Type	Location Description	Arrest	Domestic	Year
0	09/01/2018 12:01:00 AM	THEFT	RESIDENCE	False	True	2018
1	05/01/2016 12:25:00 AM	DECEPTIVE PRACTICE	NaN	False	False	2016
2	07/31/2018 01:30:00 PM	NARCOTICS	STREET	True	False	2018
"""

### 2) Check miss value
print(chicago_df.shape)
print(chicago_df.isnull().sum())# Location Description = 10758
print(chicago_df.value_counts())
"""
(7846809, 6)
Date                        0
Primary Type                0
Location Description    10758
Arrest                      0
Domestic                    0
Year                        0
dtype: int64
Date                    Primary Type       Location Description           Arrest  Domestic  Year
01/01/2008 12:01:00 AM  THEFT              RESIDENCE                      False   False     2008    90
01/01/2006 12:01:00 AM  THEFT              RESIDENCE                      False   False     2006    78
01/01/2007 12:01:00 AM  THEFT              RESIDENCE                      False   False     2007    76
01/01/2007 12:00:00 AM  THEFT              RESIDENCE                      False   False     2007    66
01/01/2003 12:01:00 AM  THEFT              RESIDENCE                      False   False     2003    58
                                                                                                    ..
05/10/2011 08:55:00 AM  BATTERY            APARTMENT                      False   True      2011     1
05/10/2011 08:52:00 PM  CRIMINAL TRESPASS  SCHOOL, PUBLIC, GROUNDS        True    False     2011     1
05/10/2011 08:50:00 PM  THEFT              STREET                         False   True      2011     1
                                                                                  False     2011     1
05/10/2011 09:00:00 PM  ASSAULT            RESIDENTIAL YARD (FRONT/BACK)  False   False     2011     1
Name: count, Length: 7216691, dtype: int64
"""

chicago_df['Location Description'] = chicago_df['Location Description'].fillna('OTHER')
print(chicago_df['Location Description'].unique())

### 3) Too many Location Desciption = Categorization
### (Categorize loactions where seem similiar)
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

### 4) Too many Primary Type = Categorization
### (Categorize loactions where seem similiar)
print(chicago_df['Primary Type'].unique())
"""['THEFT' 'DECEPTIVE PRACTICE' 'NARCOTICS' 'CRIMINAL DAMAGE'
 'OTHER OFFENSE' 'PUBLIC PEACE VIOLATION' 'BATTERY' 'CRIM SEXUAL ASSAULT'
 'BURGLARY' 'LIQUOR LAW VIOLATION' 'CRIMINAL SEXUAL ASSAULT'
 'OFFENSE INVOLVING CHILDREN' 'SEX OFFENSE' 'CRIMINAL TRESPASS'
 'WEAPONS VIOLATION' 'ROBBERY' 'MOTOR VEHICLE THEFT' 'ASSAULT' 'OBSCENITY'
 'INTERFERENCE WITH PUBLIC OFFICER' 'HUMAN TRAFFICKING' 'ARSON' 'GAMBLING'
 'PROSTITUTION' 'NON-CRIMINAL' 'INTIMIDATION' 'STALKING' 'KIDNAPPING'
 'CONCEALED CARRY LICENSE VIOLATION' 'HOMICIDE' 'OTHER NARCOTIC VIOLATION'
 'RITUALISM' 'PUBLIC INDECENCY' 'NON - CRIMINAL'
 'NON-CRIMINAL (SUBJECT SPECIFIED)' 'DOMESTIC VIOLENCE']"""

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
"""	count
Crime Group	
PROPERTY	3573715
VIOLENT	2284101
DRUGS	748906
OTHER	487284
FRAUD	449766
PUBLIC ORDER	179812
SEXUAL	66809
CHILD/FAMILY	56224
NON-CRIMINAL	192

dtype: int64"""

chicago_df.drop(columns=["Primary Type"], inplace = True)

chicago_df.head(5)
"""	Date	Arrest	Domestic	Year	Location Group	Crime Group
0	09/01/2018 12:01:00 AM	False	True	2018	RESIDENTIAL	PROPERTY
1	05/01/2016 12:25:00 AM	False	False	2016	OTHER	FRAUD
2	07/31/2018 01:30:00 PM	True	False	2018	PUBLIC	DRUGS
3	12/19/2018 04:30:00 PM	False	False	2018	PUBLIC	PROPERTY
4	02/02/2015 10:00:00 AM	False	False	2015	OTHER	FRAUD
"""

### 5) Extract month From Date Columns
chicago_df["Date"] = pd.to_datetime(chicago_df['Date'], format='%m/%d/%Y %I:%M:%S %p')

chicago_df["Month"] = chicago_df["Date"].dt.month
chicago_df.head()
"""Date	Arrest	Domestic	Year	Location Group	Crime Group	Month
0	2018-09-01 00:01:00	False	True	2018	RESIDENTIAL	PROPERTY	9
1	2016-05-01 00:25:00	False	False	2016	OTHER	FRAUD	5
2	2018-07-31 13:30:00	True	False	2018	PUBLIC	DRUGS	7
3	2018-12-19 16:30:00	False	False	2018	PUBLIC	PROPERTY	12
4	2015-02-02 10:00:00	False	False	2015	OTHER	FRAUD	2"""

chicago_df.drop(columns = ["Date"], inplace = True)
chicago_df.drop(columns = ["Year"], inplace = True)

chicago_df.head()
chicago_df.info()
"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7846809 entries, 0 to 7846808
Data columns (total 5 columns):
 #   Column          Dtype 
---  ------          ----- 
 0   Arrest          bool  
 1   Domestic        bool  
 2   Location Group  object
 3   Crime Group     object
 4   Month           int32 
dtypes: bool(2), int32(1), object(2)
memory usage: 164.6+ MB"""


import matplotlib.pyplot as plt
import pandas as pd

chicago_df['Arrest'] = chicago_df['Arrest'].astype(int)
chicago_df['Domestic'] = chicago_df['Domestic'].astype(int)

# 숫자형 컬럼만 반복
for col in chicago_df.select_dtypes(include='number'):
    chicago_df[col].plot(kind='hist', bins=30, title=col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

### 6) Encoding
### Arrest, Domestice -> Change type
### Location, Crime, Month -> One_Hot encoding
chicago_df_en = chicago_df.copy()
chicago_df_en['Arrest'] = chicago_df['Arrest'].astype(int)
chicago_df_en['Domestic'] = chicago_df['Domestic'].astype(int)

chicago_df_en = pd.get_dummies(chicago_df, columns=['Location Group','Crime Group', 'Month'])

### Change type of boolean to int : for stable model
bool_cols = chicago_df_en.select_dtypes(include='bool').columns
chicago_df_en[bool_cols] = chicago_df_en[bool_cols].astype(int)

print(chicago_df_en.dtypes)
print(chicago_df_en.shape)
print(chicago_df_en.head())

### 7) StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(chicago_df_en)
chicago_scaled = scaler.transform(chicago_df_en)

chicago_scaled = pd.DataFrame(data=chicago_scaled, columns=chicago_df_en.columns)

### 8) Check Scatter
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

## 3.  K-Means Clustering
### 1) Check k
from sklearn.utils import resample

# Resample 10K (Too Many Value)
sample_data = resample(chicago_scaled, n_samples=30000, random_state=0)

for k in range(20, 30):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(sample_data)
    score = silhouette_score(sample_data, labels)
    print(f'k={k}, silhouette : {score:.4f}')
"""k=20, silhouette : 0.2944
k=21, silhouette : 0.2863
k=22, silhouette : 0.2694
k=23, silhouette : 0.2901
k=24, silhouette : 0.2944
k=25, silhouette : 0.2860
k=26, silhouette : 0.2823
k=27, silhouette : 0.2769
k=28, silhouette : 0.2785
k=29, silhouette : 0.2575"""
# k = 24

### 2) Optimized K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

kmeans = KMeans(n_clusters = 24, init = 'k-means++', max_iter=300, random_state=0)
kmeans_labels = kmeans.fit_predict(sample_data)

### 3) Analyze Clustering
clustered_df = pd.DataFrame(sample_data, columns = chicago_scaled.columns)
clustered_df['cluster'] = kmeans_labels

#### 1. Extract Column
location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if "Crime Group_" in col]
month_col = [col for col in clustered_df.columns if "Month_" in col]

#### 2. Mean
location_summary = clustered_df.groupby('cluster')[location_col].mean()
top_lo = location_summary.idxmax(axis=1)

crime_summary = clustered_df.groupby('cluster')[crime_col].mean()
top_crime = crime_summary.idxmax(axis=1)

month_summary = clustered_df.groupby('cluster')[month_col].mean()
top_mon = month_summary.idxmax(axis=1)

#### 3. DataFrame
cluster_summary = pd.DataFrame({
    "Top Location" : top_lo,
    "Top_Crime" : top_crime,
    "Top_Month" : top_mon
})

print(cluster_summary)
"""                      Top Location                 Top_Crime Top_Month
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
23       Location Group_RECREATIONAL      Crime Group_PROPERTY   Month_8"""

### 4) Visualization Clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(clustered_df.drop('cluster', axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clustered_df['cluster'], cmap='tab10', s=10, alpha=0.6)
plt.title('K-means Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

## 4.Mean Shift Clustering
### 1) Using estimate_bandwidth()
from sklearn.cluster import estimate_bandwidth
from sklearn.utils import resample

sample_data = resample(chicago_scaled, n_samples=10000, random_state=42)

bandwidth = estimate_bandwidth(sample_data, quantile=0.1)
print(f"bandwidth : {bandwidth:.4f}")
# bandwidth : 6.0248

### 2) Mean Shift Clustering
from sklearn.cluster import MeanShift

meanshift = MeanShift(bandwidth=6)
cluster_labels = meanshift.fit_predict(sample_data)

print(np.unique(cluster_labels))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

### 3) Check Values in Mean_Shift Clustering
clustered_df = pd.DataFrame(sample_data, columns = chicago_scaled.columns)
clustered_df['cluster'] = cluster_labels

#### 1. Extract Column
location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if "Crime Group_" in col]
month_col = [col for col in clustered_df.columns if "Month_" in col]

#### 2. Mean
location_summary = clustered_df.groupby('cluster')[location_col].mean()
top_lo = location_summary.idxmax(axis=1)

crime_summary = clustered_df.groupby('cluster')[crime_col].mean()
top_crime = crime_summary.idxmax(axis=1)

month_summary = clustered_df.groupby('cluster')[month_col].mean()
top_mon = month_summary.idxmax(axis=1)

#### 3. DataFrame
cluster_summary = pd.DataFrame({
    "Top Location" : top_lo,
    "Top_Crime" : top_crime,
    "Top_Month" : top_mon
})

print(cluster_summary)
"""                       Top Location                 Top_Crime Top_Month
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
15          Location Group_EDUCATION  Crime Group_CHILD/FAMILY  Month_10"""

### 4) Visualization Clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(clustered_df.drop('cluster', axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clustered_df['cluster'], cmap='tab10', s=10, alpha=0.6)
plt.title('Meanshift Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

## 3. GMM Clustering
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample

sample_data = resample(chicago_scaled, n_samples=10000, random_state=42)

gmm = GaussianMixture(n_components = 8, random_state=0).fit(sample_data)
gmm_cluster_labels = gmm.predict(sample_data)

### 1) Check Values in GMM Clustering
clustered_df = pd.DataFrame(sample_data, columns = chicago_scaled.columns)
clustered_df['cluster'] = gmm_cluster_labels

#### 1. Extract Column
location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if "Crime Group_" in col]
month_col = [col for col in clustered_df.columns if "Month_" in col]

#### 2. Mean
location_summary = clustered_df.groupby('cluster')[location_col].mean()
top_lo = location_summary.idxmax(axis=1)

crime_summary = clustered_df.groupby('cluster')[crime_col].mean()
top_crime = crime_summary.idxmax(axis=1)

month_summary = clustered_df.groupby('cluster')[month_col].mean()
top_mon = month_summary.idxmax(axis=1)

#### 3. DataFrame
cluster_summary = pd.DataFrame({
    "Top Location" : top_lo,
    "Top_Crime" : top_crime,
    "Top_Month" : top_mon
})

print(cluster_summary)
"""                      Top Location             Top_Crime Top_Month
cluster                                                            
0         Location Group_COMMERCIAL     Crime Group_FRAUD   Month_1
1             Location Group_PUBLIC     Crime Group_DRUGS   Month_5
2            Location Group_MEDICAL     Crime Group_OTHER   Month_5
3        Location Group_RESIDENTIAL   Crime Group_VIOLENT  Month_10
4         Location Group_INDUSTRIAL  Crime Group_PROPERTY  Month_12
5          Location Group_EDUCATION     Crime Group_OTHER   Month_4
6         Location Group_COMMERCIAL  Crime Group_PROPERTY   Month_6
7             Location Group_PUBLIC     Crime Group_DRUGS  Month_11
"""

### 2) Analyze GMM CLuster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(clustered_df.drop('cluster', axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clustered_df['cluster'], cmap='tab10', s=10, alpha=0.6)
plt.title('GMM Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

## 4. DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.6, min_samples = 8, metric = 'euclidean')
dbscna_labels = dbscan.fit_predict(sample_data)

### 1) Check Values in DBSCAN Clustering
clustered_df = pd.DataFrame(sample_data, columns = chicago_scaled.columns)
clustered_df['cluster'] = dbscna_labels

#### 1. Extract Column
location_col = [col for col in clustered_df.columns if 'Location Group_' in col]
crime_col = [col for col in clustered_df.columns if "Crime Group_" in col]
month_col = [col for col in clustered_df.columns if "Month_" in col]

#### 2. Mean
location_summary = clustered_df.groupby('cluster')[location_col].mean()
top_lo = location_summary.idxmax(axis=1)

crime_summary = clustered_df.groupby('cluster')[crime_col].mean()
top_crime = crime_summary.idxmax(axis=1)

month_summary = clustered_df.groupby('cluster')[month_col].mean()
top_mon = month_summary.idxmax(axis=1)

#### 3. DataFrame
cluster_summary = pd.DataFrame({
    "Top Location" : top_lo,
    "Top_Crime" : top_crime,
    "Top_Month" : top_mon
})

print(cluster_summary)
"""                        Top Location                 Top_Crime Top_Month
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
 8          Location Group_EDUCATION  Crime Group_PUBLIC ORDER  Month_10"""

### 2) Analyze DBSCAN CLuster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(clustered_df.drop('cluster', axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clustered_df['cluster'], cmap='tab10', s=10, alpha=0.6)
plt.title('DBSCAN Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()