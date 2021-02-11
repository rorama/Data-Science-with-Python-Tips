#!/usr/bin/env python
# coding: utf-8

# # <font color=blue><b>Lesson 1: DataFrame basics</b></font>

# In[808]:

## editing git checkout -b data_science_test2

import pandas as pd


# In[809]:


a = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]}, index=["A", "B"])
a


# In[810]:


b = pd.Series([4, 5], index=["a", "b"], name="Product")
b


# # <font color=blue><b>Lesson 2: DataFrame summary - read_csv, describe(), apply, lambda</b></font>

# In[910]:


contacts = pd.read_csv("D:/Temp/contacts.csv")

#use contacts.describe() to print descriptive statistics of numeric columns (noten that NaN is treated to be numeric)
#contacts.describe() #all columns
contacts.describe(include=['object']) #only columns with object type
contacts.describe(include=['float']) #only columns with float type

# create a function #or use lambda and ternary form below
def Site_Short(x):
    if x == 'Mulago':
        return 'Mul'
    if x == 'MKC':
        return 'MKC'

# apply the function to the gender column and create a new column
#contacts['Site_New'] = contacts['Site'].apply(Site_Short)

#lambda: anonymous one-line function
contacts['Site_New'] = contacts['Site'].apply(lambda x: 'MUL' if x=='Mulago' else ('MKC' if x=='MKC' else ''))

contacts.head(7)


# # <font color=blue><b>Lesson 3: DataFrame shape, iloc, loc</b></font>

# In[812]:


contacts.shape #dimensions


# In[1008]:


contacts.Section.head() #dot notation
contacts['Section'].head() #array notation??


# In[814]:


contacts['Section'].head()


# In[815]:


#Difference between loc (label-based selection) and iloc (integer-based selection) 
#iloc in Python. You can use iloc in Python for selection. It is integer-location based and helps you select by the position. ...
#loc in Pandas. You can use loc in Pandas to access multiple rows and columns by using labels; however, you can use it with a boolean array as well (for filtering rows by a condition). 

contacts.iloc[:10, [0,4]].head()


# In[816]:


contacts.iloc[-10:, [0, 2, -2, -1]].head()


# In[817]:


contacts.loc[:, ["Section", "Site"]].head()


# In[1047]:


#display Site, Department and Section for rows that pass the condition - very good for filtering
#to negate condition, use ~, eg "not in"  would be ~contacts.Department.isin(['SPDT', 'Training'])
#contacts.loc[(contacts.Site == 'MKC') & (contacts.Department.isin(['SPDT', 'Training'])) & (contacts['Project Name'].notnull()), ['Site']]
contacts.loc[(contacts.Site == 'MKC') & (contacts.Department.isin(['SPDT', 'Training'])), ['Site', 'Department', 'Section']]


# # <font color=blue><b>Lesson 4: DataFrame groupby</b></font>

# In[819]:


contacts.groupby('Site').Section.min()


# In[1160]:


#contacts.groupby(['Department']).Department.value_counts()
contacts.groupby(['Department']).salary.sum()/contacts.Department.value_counts().sort_values(ascending = False)


# In[1155]:


#contacts.groupby(['Section', 'Site']).apply(lambda df: df.iloc[0]).head(20)
contacts.groupby(['Section', 'Site']).apply(lambda df: df.salary.iloc[0]).head(20)


# In[868]:


#for each Department, show pecentage by Site
#contacts.groupby(['Department', 'Site']).Site.agg('count')
site_dept_count = contacts.groupby(['Department', 'Site']).agg({'Site': 'count'})
dept_count = contacts.groupby(['Department']).agg('count')

# divide the gender_ocup per the occup_count and multiply per 100
dept_site = site_dept_count.div(dept_count, level = "Department") * 100

# present all rows from the 'site column'
dept_site.loc[:, 'Site']


# In[822]:


#contacts = pd.read_csv("D:/Temp/contacts.csv")


# In[1161]:


contacts['salary']=5
contacts['Section']=contacts['Section'].str.replace('/', '-') #use .str.replace to replace the whole field
contacts['date_created']='2021-02-08'


# In[1162]:


contacts.head()


# In[1163]:


contacts.mean()


# In[1164]:


contacts.groupby(['Site', 'Section']).salary.sum()


# In[1167]:


#idxmin - by default, it returns the index for the minimum value in each column
contacts.groupby(['Site', 'Section']).apply(lambda df: df.loc[df.salary.idxmin()]) #least paid in each Site/Section
#contacts.groupby(['Site', 'Section']).apply(lambda df: df) #list all


# In[1168]:


contacts.groupby(['Site', 'Department']).apply(lambda df: df.loc[df.salary.idxmax()])


# In[1169]:


contacts.groupby(['Site', 'Department']).apply(lambda df: df.loc[df.salary.idxmax()]).sort_values(by=['Section', 'Staff Name'], ascending=False, na_position='first')


# In[1170]:


contacts.groupby(['Site', 'Department']).salary.agg(['sum', 'mean', 'median', 'count']) #.salary.sum() #.salary.agg([sum]) #(by='Section', key=lambda col: col.str.lower())


# # <font color=blue><b>Lesson 5: DataFrame column rename, concat, merge</b></font>

# In[1184]:


contacts.rename(columns={'salary': 'Salary'}).head()


# In[832]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
'B': ['B0', 'B1', 'B2', 'B3'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']},
index=[0, 1, 2, 3])


# In[833]:


df2 = pd.DataFrame({'A': ['A20', 'A21', 'A22', 'A23'],
'B': ['B0', 'B1', 'B2', 'B3'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']},
index=[4, 5, 6, 7])


# In[834]:


df3 = pd.concat([df1, df2]).drop_duplicates() #can drop duplicates
df3


# In[835]:


df3 = pd.concat([df1, df2], keys=['xx', 'yy'])
df3


# In[836]:


df3.loc['yy']


# In[837]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
'B': ['B0', 'B1', 'B2', 'B3'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']},
index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A0', 'A21', 'A22', 'A23'],
'B': ['Bx0', 'B1', 'B2', 'B3'],
'C': ['Cx0', 'C1', 'C2', 'C3'],
'D': ['Dx0', 'D1', 'D2', 'D3']},
index=[4, 5, 6, 7])


# In[838]:


df3 = pd.merge(df1, df2, on='A', how='left') #, left_index=True, right_index=True,)
df3


# # <font color=blue><b>Lesson 6: DataFrame matplotlib plot</b></font>

# In[839]:


import matplotlib.pyplot as plt
import numpy as np


# In[840]:


plt.close('all')
ts = pd.Series(np.random.randn(1000),
    index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()

ts.plot()


# # <font color=blue><b>Lesson 7: Mapping with GeoPandas</b></font>

# In[877]:


import geopandas as gpd
#import matplotlib.pylpot as plt #already imported up there

#from mpl_toolkits.axes_grid1 import make_axes_locatable

uganda = gpd.read_file('D:/SMC Operational Center/data/Uganda_districts2010.shp')
uganda_lakes_rivers = gpd.read_file('D:/SMC Operational Center/data/Uganda_lakes_rivers2005.shp')

#uganda.plot(cmap='jet', figsize=(10, 10))
#type(uganda)#geopandas.geodataframe.GeoDataFrame
#uganda.crs #see the coordinate reference system

uganda_hiv = pd.read_csv('D:/Temp/Ugands_HIV.csv')
uganda_hiv.fillna(1, inplace=True)
#uganda_hiv[uganda_hiv['hiv_rate'] < 8] = 0
uganda_hiv.loc[(uganda_hiv.hiv_rate < 5), 'hiv_rate'] = 0


uganda_hiv['hiv_notes'] = [str(x*100) + ' people' for x in uganda_hiv['hiv_rate']]


#merge uganda with uganda_hiv
uganda = uganda.merge(uganda_hiv, on='DNAME_2010')
#uganda[uganda['hiv_rate'] < 0] = None


ugandaHF = gpd.read_file('D:/SMC Operational Center/data/UgandaHF.shp')
#ugandaHF.plot(cmap='jet', column='level', figsize=(10, 10))

#fig, ax1 = plt.subplots(1, figsize=(15, 15))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20), subplot_kw=dict(aspect="equal"))

#divider = make_axes_locatable(ax1)

#fig.suptitle('Map of Uganda', fontsize=24, color='blue')
ax1.set_title('Map of Uganda showing HIV Rates \n(Hypothetical Data)', fontsize=20, color='blue')
#ax1.set_xlabel('xlabel')
#ax1.set_ylabel('ylabel')
#ax1.set_axis_off()
#plt.xticks([])
#plt.yticks([])
ax1.set_axis_off()

# Adjust legend location
leg = ax1.get_legend()
#leg.set_bbox_to_anchor((1.15,0.5))

#ax1.set_axis_off()

#ax1.text(0, 0, r'an equation: $E=mc^2$', fontsize=15)

centroids = uganda.copy()
centroids.geometry = uganda.centroid
centroids['size'] = centroids['hiv_rate'] * 100  # to get reasonable plotable number

#uganda.plot(ax=ax1
uganda[uganda['hiv_rate'] >= 0].plot(ax=ax1
            , cmap='OrRd' #'YlOrBr' #'Set1' #'Dark2' #'prism' #'plasma' #'Pastel1'
            #, alpha=0.5
            #, facecolor='k'
            , edgecolor='black'
            , column='hiv_rate'
            , legend=True                                     
            , legend_kwds={'label': "HIV Rate", 'orientation': 'horizontal'}
            #, missing_kwds={
            #    "color": "lightgrey",
            #    "edgecolor": "red",
            #    "hatch": "///",
            #    "label": "Missing values"
            #}
            , linewidth=1
            , zorder=0                                     
)

centroids.plot(ax=ax1, column='hiv_rate', legend=True, categorical=True, legend_kwds={}, marker='p', markersize='size')

#uganda.boundary.plot(ax=ax1)
#uganda.apply(lambda x: ax1.annotate(s=x.DNAME_2010, xy=x.geometry.centroid.coords[0], ha='left', fontsize=6, color='black', horizontalalignment='right', verticalalignment='center', textcoords='data'),axis=1)
uganda_lakes_rivers.plot(ax=ax1, color='lightsteelblue', legend=True, alpha=0.7)
ugandaHF.plot(ax=ax1, cmap='jet', marker='P', markersize=10, column='level', legend=True, categorical=True, zorder=1)
#ax1.plot(activity, dog, label="dog")
#ax1.plot(activity, cat, label="cat")
#ax1.set_label('Label via method')
#ax1.legend()


ax1.legend([red_dot, (red_dot, white_cross), red_dot], ['Districts', 'Facilities', 'Water bodies'],
          title="Legend",
          loc="upper left",
          bbox_to_anchor=(1, 0, 0.5, 1))

#ax1.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])


#legend_elements =[]
#ax1.legend(handles=legend_elements, loc='lower right', fontsize=20, title_fontsize=20);


#uganda[uganda['hiv_rate'] >= 0].bar(ax=ax2)
ax2.pie(uganda.loc[(uganda.hiv_rate > 5), 'hiv_rate'], labels=uganda.loc[(uganda.hiv_rate > 5), 'DNAME_2010'], autopct='%1.1f%%', shadow=True, startangle=90)
#ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax3.bar(uganda.loc[(uganda.hiv_rate > 5), 'DNAME_2010'], uganda.loc[(uganda.hiv_rate > 5), 'hiv_rate'] , 0.5, color='r')
ax4.plot(uganda.loc[(uganda.hiv_rate > 5), 'DNAME_2010'], uganda.loc[(uganda.hiv_rate > 5), 'hiv_rate'] , 0.5, color='b')
#ax4.plot(uganda['hiv_rate'])

#ax1.legend(['xx1', 'xx2'])
#ax2.set_axis_off()

# Create a output path for the data
out = r"D:/Temp/GeoPandas/uganda.shp"
# Select first 50 rows
selection = uganda[0:50] #uganda #uganda[0:50]
# Write those rows into a new Shapefile (the default output file format is Shapefile)
selection.to_file(out)

#plt.figure(dpi=4000)
plt.savefig('D:/Temp/uganda.png', dpi=200) 

uganda_html = uganda.to_html(index_names=False) #to html table


# # <font color=blue><b>Lesson 8: Pie charts</b></font>

# In[902]:


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["375 g flour",
          "75 g sugar",
          "250 g butter",
          "300 g berries"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

#Adding wedgeprops=dict(width=0.5) changes the pie into donut
#wedges, texts, autotexts = ax.pie(data, wedgeprops=dict(width=0.5), autopct=lambda pct: func(pct, data),
#                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("Matplotlib bakery: A pie")

plt.show()


# # <font color=blue><b>Lesson 9: Donut charts - just another piechart by adding wedgeprops</b></font>

# In[899]:


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["225 g flour",
          "90 g sugar",
          "1 egg",
          "60 g butter",
          "100 ml milk",
          "1/2 package of yeast"]

data = [225, 90, 50, 60, 100, 5]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square, pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Matplotlib bakery: A donut")

plt.show()


# # <font color=blue><b>Lesson 10: Reading from Database, dropping columns, frequency counts</b></font>

# In[906]:


from sqlalchemy import create_engine
import pymysql #install case-sensitively using: pip install PyMySQL
#sqlEngine       = create_engine('mysql+pymysql://root:@127.0.0.1', pool_recycle=3600)

dbConnection = pymysql.connect(host='localhost', port=3306, user='grails', password='server', db='nomad')
equipment = pd.read_sql("select * from nomad.equipment", dbConnection);


good_columns = equipment._get_numeric_data().dropna(axis=1) #remove columns with non-numeric and missing values (NA, Nan, etc)

#good_columns.shape #(556, 6)
#equipment.shape #(556, 19)

#equipment['recommended_health_center2'].unique()
#equipment.date_created.value_counts() #frequency counts
#equipment['date_created'].value_counts() #frequency counts
#equipment


# # <font color=blue><b>Lesson 11: Counting missing values</b></font>

# In[882]:


equipment.fillna(equipment.mean(), inplace=True)
equipment.dtypes

# Define function
def missing(x):
    return sum(x.isnull())

# Apply per row
print('Missing values per row')
equipment.apply(missing, axis = 1) #.head()

# Apply per column
print('Missing values per column')
equipment.apply(missing, axis = 0) #.head()


# In[846]:


equipment.head(2)


# # <font color=blue><b>Lesson 12: Data exploration - seaborn</b></font>

# In[847]:


#data exploration - see correlation among then different variables
import seaborn as sns
#import matplotlib.pyplot as plt
sns.pairplot(equipment[["recommended_general_hospital", "recommended_health_center2", "recommended_health_center3", "recommended_health_center4"]])
plt.show()


# # <font color=blue><b>Lesson 13: Machine learning - KMeans</b></font>

# In[848]:


from sklearn.cluster import KMeans #the main Python machine learning package, scikit-learn
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = equipment._get_numeric_data().dropna(axis=1) #remove columns with non-numeric and missing values (NA, Nan, etc)
kmeans_model.fit(good_columns) #fit a k-means clustering model
labels = kmeans_model.labels_ #and get our cluster labels


# In[849]:


from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()


# In[850]:


# Splitting Data into Training and Testing Sets
# for supervised machine learning, it’s a good idea to split the data into training and testing sets so we don’t overfit

train = equipment.sample(frac=0.8, random_state=1) #returns size (445, 19)
test = equipment.loc[~equipment.index.isin(train.index)] #returns size (111, 19)


# # <font color=blue><b>Lesson 14: Machine learning - Linear Regression</b></font>

# In[851]:


# Univariate Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[["recommended_health_center2"]], train["recommended_health_center3"])
predictions = lr.predict(test[["recommended_health_center2"]])


# In[852]:


# Calculating Summary Statistics for the Model
import statsmodels.formula.api as sm
model = sm.ols(formula='recommended_health_center3 ~ recommended_health_center2', data=train)
fitted = model.fit()
fitted.summary()


# # <font color=blue><b>Lesson 15: Machine learning - Random Forest</b></font>

# In[853]:


# Fit a random forest model
from sklearn.ensemble import RandomForestRegressor
predictor_columns = ["recommended_health_center2", "recommended_health_center4", "recommended_general_hospital"]
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[predictor_columns], train["recommended_health_center3"])
predictions = rf.predict(test[predictor_columns])


# In[854]:


# Calculating Error
# Now that we’ve fit two models, let’s calculate error in R and Python. We’ll use MSE.

from sklearn.metrics import mean_squared_error
mean_squared_error(test["recommended_health_center3"], predictions)


# # <font color=blue><b>Lesson 16: Machine learning - Web scraping</b></font>

# In[855]:


# web scraping
# import requests # The requests library will make a GET request to a web server, which will download the HTML contents of a given web page for us. There are several different types of requests we can make using requests, of which GET is just one
url = "http://www.basketball-reference.com/boxscores/201506140GSW.html"
data = requests.get(url).content
data


# In[856]:


from bs4 import BeautifulSoup
soup.prettify()
list(soup.children)
[type(item) for item in list(soup.children)] #list-comprehension, concise way to create new lists
html = list(soup.children)[1] #html is element 3 in list(soup.children)

soup = BeautifulSoup(data, 'html.parser')
soup.find_all('p')
soup.find_all('table') #[0].get_text()
len(soup.find_all('tr'))


# In[857]:


from bs4 import BeautifulSoup # we use BeautifulSoup, the most commonly used web scraping package
import re
soup = BeautifulSoup(data, 'html.parser')
box_scores = []
for tag in soup.find_all(id=re.compile("[A-Z]{3,}_basic")):
    rows = []
    for i, row in enumerate(tag.find_all("tr")):
        if i == 0:
            continue
        elif i == 1:
            tag = "th"
        else:
            tag = "td"
        row_data = [item.get_text() for item in row.find_all(tag)] #list comprehension
        rows.append(row_data)
        box_scores.append(rows)
        
box_scores


# In[858]:


#import requests
#from bs4 import BeautifulSoup
import random
 
text = 'python'
url = 'https://google.com/search?q=' + text
A = ("Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
       )
 
Agent = A[random.randrange(len(A))]
 
headers = {'user-agent': Agent}
r = requests.get(url, headers=headers)
 
soup = BeautifulSoup(r.text, 'lxml') #The ‘lxml‘ package must be installed for the below code to work.
for info in soup.find_all('h3'):
    print(info.text)
    print('#######')

r.content
#soup.find_all('h3')    


# In[859]:


xx='$123457'
#float(xx[1:4])
float(xx.replace('$', ''))


# # <font color=blue><b>Lesson 17: Apply, drop, filter, lambda</b></font>

# In[1110]:


#contacts
#contacts[ #-2:, :3] #not work

#apply is used to apply a function along an axis of the DataFrame.

contacts['AllColumns'] = contacts[contacts.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)

contacts.loc['Row Total'] = contacts.loc[:,'Column Total'] = 0
contacts.loc['Row Total'] = contacts.sum(numeric_only=True, axis=0)
contacts.loc[:,'Column Total'] = contacts.sum(numeric_only=True, axis=1)
#contacts.loc['Row Total'].fillna('xxx')
#contacts.loc['Row Total'].isnull().values.any()
#contacts.loc[:,'Column Total'].isnull().values.any()

#drop columns
#contacts.drop(columns=['Column1', 'Column2'], inplace=True)

#drop row by index
#contacts.drop(['Row1'], inplace=True) #inplace=True is same as assigning df=df.drop()...

#drop all rows with Department equals Training
contacts.drop(contacts[contacts['Department'] == 'Training'].index, inplace = True) 
contacts['Department'].unique()
#could have used vectorization operation: contacts[contacts['Department'] != 'Training'] to achieve the same filtering

contacts[contacts['Department'] != 'Training'].loc[:, 'Site'] #display Site only


#contacts.groupby(["Department", "Section"]).sum() #displays totals only
#contacts.groupby(["Department", "Section"]).apply(lambda df: df.loc[df.salary.idxmin()])
grouped = contacts.groupby(["Department", "Section"])

#for name,group in grouped:
#for name,group in grouped:
#    print (name)
#    print (group)

#print (grouped.get_group("ED's Office", "ED's Office"))

#print (grouped['salary'].agg([np.mean, np.median, np.sum]))

grouped.filter(lambda x: len(x) >= 60)

#contacts.Salary.filter(lambda x: len(x) >= 60) #not work

#contacts.dtypes #returns a series with data type of each column


# TO convert the date to datetime64 
contacts['date_created'] = pd.to_datetime(contacts['date_created'], format='%Y-%m-%d') 
  
# Filter data between two dates using query function
filtered_df = contacts.query("date_created == '2020-02-08' and date_created == '2021-02-08'") 
filtered_df

#filtered_df1 = contacts.query('Site == "MKC" or Department == "PCT"') #.query("Department.isin(['SPDT', 'Training'])") 
#filtered_df1.loc[:, ['Department', 'Site', 'AllColumns']]
filtered_df1 = contacts.query('Site == "MKC" or Department == "PCT"').loc[:, ['Department', 'Site', 'AllColumns']]
filtered_df1.loc[contacts.Site == 'MKC', 'Department']


# # <font color=blue><b>Lesson 18: Reading from Excel, reset index, drop na</b></font>

# In[965]:


staffpickup = pd.ExcelFile("D:\Temp\staffpickup.xlsx")
staffpickup = staffpickup.parse("March-2020", header=3) #[2:].reset_index()
staffpickup.head()


# In[966]:


#staffpickup.dropna() #Drop the rows where at least one element is missing.
#staffpickup.dropna(axis='columns') #Drop the columns where at least one element is missing.
#staffpickup.drop(columns=['index'], inplace=True)
staffpickup.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True) #drop rows with all values missing 
#staffpickup.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True) #drop rows with all values missing 
staffpickup.drop(columns=['Unnamed: 5'], inplace=True)
staffpickup.reset_index(drop=True, inplace=True) #drop existing index
staffpickup


# # <font color=blue><b>Lesson 19: Natural Language Processing</b></font>

# In[862]:


# install nltk: pip install nltk

import nltk
#nltk.download('punkt') #do this once only

text="""Hello Mr. Smith, how are you doing today? The weather is great connect, and city is awesome connected.
The sky is pinkish-blue connecting. You shouldn't eat cardboard"""

#text=contacts['AllColumns'] #convert to string

#f = open("D:/Temp/demofile.txt", "r")
#text=f.read()


from nltk.tokenize import sent_tokenize
tokenized_sent=sent_tokenize(text) #Sentence tokenizer breaks text paragraph into sentences
tokenized_sent #['Hello Mr. Smith, how are you doing today?', 'The weather is great, and city is awesome.', 'The sky is pinkish-blue.', "You shouldn't eat cardboard"]

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)#Word tokenizer breaks text paragraph into words.
tokenized_word #['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', ',', 'and', 'city', 'is', 'awesome', '.', 'The', 'sky', 'is', 'pinkish-blue', '.', 'You', 'should', "n't", 'eat', 'cardboard']


from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
fdist #FreqDist({'is': 3, ',': 2, 'The': 2, '.': 2, 'Hello': 1, 'Mr.': 1, 'Smith': 1, 'how': 1, 'are': 1, 'you': 1, ...})

fdist.most_common(2) #[('is', 3), (',', 2)]

# Frequency Distribution Plot
#import matplotlib.pyplot as plt
#fdist.plot(30,cumulative=False)
#plt.show()

def myPlot(myDist):
    import matplotlib.pyplot as plt
    myDist.plot(30,cumulative=False)
    plt.show()    

myPlot(fdist)

#Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.
#from nltk.corpus import stopwords #run only once

#nltk.download('stopwords') #run only once
stop_words=set(stopwords.words("english"))
stop_words


#removing stopwords
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w.upper()) #convert to uppercase

freqFilteredWord = FreqDist(filtered_word)
        
myPlot(freqFilteredWord)

#print("Tokenized Word:",tokenized_word)
#print("Filterd Word:",filtered_word)


#Lexicon normalization considers another type of noise in the text. For example, connection, connected, connecting word reduce to a common word "connect". It reduces derivationally related forms of a word to a common root word.

# Stemming: Stemming is a process of linguistic normalization, which reduces words to their word root word or chops off the derivational affixes. For example, connection, connected, connecting word reduce to a common word "connect".
from nltk.stem import PorterStemmer
#from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_word:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_word)
print("Stemmed Sentence:",stemmed_words)



#freqFilteredWord


# # <font color=blue><b>Lesson 20: Crosstab and Pivot</b></font>

# In[974]:


contacts['Distance'] = contacts['Site'].apply(lambda x: 110 if x=='Mulago' else 220)
contacts.head()


# In[985]:


ctab = pd.crosstab(contacts['Department'], contacts['Site'])
ctab.T #T for Transpose
ctab.apply(lambda r: (r/len(contacts))*100, axis=1) #percentages
ctab.T #T for Transpose


# In[989]:


pd.pivot_table(contacts, index=['Site', 'Department'], values='Distance', aggfunc=['mean', 'median', 'sum'])


# In[998]:


pd.pivot_table(contacts, index='Site', columns=['Department', 'Section'], values='Distance', aggfunc='mean')

