import pandas  as pd
import numpy as np
import json
from scipy import spatial

movies = pd.read_csv('/Users/nikhil/Data/ML_examples/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/nikhil/Data/ML_examples/tmdb_5000_credits.csv')


#converting json to string for genres, keywords, cast and director
i=0
movies['genres'] = movies['genres'].apply(json.loads)
for i in range (0,movies.shape[0]):
    Name = []
    j=0
    for j in range (0,np.shape(movies['genres'][i])[0]):
        Name.append(movies['genres'][i][j]['name'])
    movies.loc[i,'genres'] = str(Name)
    
i=0
movies['keywords'] = movies['keywords'].apply(json.loads)
for i in range (0,movies.shape[0]):
    Key = []
    j=0
    for j in range (0,np.shape(movies['keywords'][i])[0]):
        Key.append(movies['keywords'][i][j]['name'])
    movies.loc[i,'keywords'] = str(Key)

i=0
credits['cast'] = credits['cast'].apply(json.loads)
for i in range (0,credits.shape[0]):
    Cst = []
    j=0
    for j in range (0,np.shape(credits['cast'][i])[0]):
        Cst.append(credits['cast'][i][j]['name'])
    credits.loc[i,'cast'] = str(Cst)
    
credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'},inplace=True)


#Merging the two files
movies = movies.merge(credits)
movies = movies[['id','original_title','genres','cast','vote_average','director','keywords']]




#separating strings with multiple entries
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')



#\\\\\\\\\\\\\\\\\
#Replace director name with empty if none
i=0
for i in range (0,len(movies)):
    a = movies['director'][i]
    if a is None:
       a = ''




#Preprocessing genres, keywords, cast and director
#\\\\\\\\\\\\\\\\\
#Loop to list genres in alphabetical order
i=0
for i in range (0,len(movies)):
    sgList = []
    sgList = movies['genres'][i]
    sgList.sort()
    movies.loc[i,'genres']=str(sgList)
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')


#\\\\\\\\\\\\\\\\\
#Loop to select just the first keyword
i=0
for i in range (0,len(movies)):
    kylist = []
    kylist = movies['keywords'][i][0]
    movies.loc[i,'keywords']=str(kylist)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')


#\\\\\\\\\\\\\\\\\
#Loop to select the first cast member
i=0
for i in range (0,len(movies)):
    cList = []
    cList = movies['cast'][i][:1]
    movies.loc[i,'cast']=str(cList)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')
#Loop to list cast members in alphabetical order
i=0
for i in range (0,len(movies)):
    scList = []
    scList = movies['cast'][i]
    scList.sort()
    movies.loc[i,'cast']=str(scList)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')




#Get unique genres,keywords,cast and director

#Loop to get unique genres
gmList=[]
i=0
for i in range (0,len(movies)):
    j=0
    for j in range (0,np.shape(movies['genres'][i])[0]):
        gn = movies['genres'][i][j]
        if gn not in gmList:
           gmList.append(gn)
           
#Loop to get unique key words
kmList=[]
i=0
for i in range (0,len(movies)):
        ky = movies['keywords'][i][0]
        if ky not in kmList:
           kmList.append(ky)
           
#Loop to get unique cast members
cmList=[]
i=0
for i in range (0,len(movies)):
        cst = movies['cast'][i][0]
        if cst not in cmList:
           cmList.append(cst)

#Loop to get unique directors
dmList=[]
i=0
for i in range (0,len(movies)):
    d = movies['director'][i]
    if d not in dmList:
           dmList.append(d)


#Generate OHE for genres, keywords, cast, director
#OHE for genres
OHE_genre = np.zeros((len(movies),len(gmList)))
g=0
for g in range (0,len(gmList)):
    i = 0
    for i in range (0,len(movies)):
        j=0
        for j in range (0,np.shape(movies['genres'][i])[0]):
            if gmList[g] == movies['genres'][i][j]:
               OHE_genre[i,g] =1
               
#OHE for keywords
OHE_key = np.zeros((len(movies),len(kmList)))
g=0
for g in range (0,len(kmList)):
    i = 0
    for i in range (0,len(movies)):
        if kmList[g] == movies['keywords'][i][0]:
               OHE_key[i,g] =1

#OHE for cast
OHE_cast = np.zeros((len(movies),len(cmList)))
g=0
for g in range (0,len(cmList)):
    i = 0
    for i in range (0,len(movies)):
        if cmList[g] == movies['cast'][i][0]:
               OHE_cast[i,g] =1

#OHE for director
OHE_dir = np.zeros((len(movies),len(dmList)))
g=0
for g in range (0,len(dmList)):
    i = 0
    for i in range (0,len(movies)):
        if dmList[g] == movies['director'][i]:
               OHE_dir[i,g] =1
               


#Ask for input and run K-NN for predicting rating with K=5 and list recommended movies with K=50 and rating above 6.8
des='no'
while des != 'yes':
 Input = str(input("Enter your movie: "))
 Title = np.array(movies['original_title'])
 Keys = np.array(movies['keywords'])
 Gn = np.array(movies['genres'])
 Rating = np.array(movies['vote_average'])
 Director = np.array(movies['director'])
 Cast = np.array(movies['cast'])
 if np.shape(np.where(Title==Input))[1] == 0:
    print("The name you entered is incorrect. Please enter the correct name.")
 else:
    P_index = np.where(Title==Input)[0][0]
    ALL_OHE = np.concatenate((OHE_genre,OHE_key,OHE_cast,OHE_dir),axis=1)
    X_p = ALL_OHE[P_index]
    dist = np.zeros(len(OHE_genre))
    for j in range (0,len(OHE_genre)):
        dist[j] = spatial.distance.cosine(X_p,ALL_OHE[j,:])
        min_index = np.argsort(dist)[:50]
        min_index_pred = np.argsort(dist)[:5]

    new_min_index_all = np.delete(min_index,np.where(min_index==P_index))
    new_min_index = new_min_index_all[np.where(Rating[new_min_index_all]>6.8)]
    new_min_index_pred = np.delete(min_index_pred,np.where(min_index_pred==P_index))
    Pred_rating = np.sum(Rating[new_min_index_pred])/len(new_min_index_pred)
    Actual_rating = Rating[P_index]
    print('\n')
    print('You watched : '+str(Title[P_index]))
    print('Genre: '+str(Gn[P_index]))
    print('Keywords: '+str(Keys[P_index][0]))
    print('Director: '+str(Director[P_index]))
    print('Lead Cast: '+str(Cast[P_index][0]))
    print('Actual Rating: '+str(Rating[P_index]))
    print('Predicted Rating: ' + str(Pred_rating))
    print('\n')
    if Rating[P_index] < 6.8:
       print('The ratings for this movie are too low! Here are some suggestions above 6.8 rating:')
    else:
       print('This movie has good ratings! You might also like:')
    print('\n')

    k=0
    for k in range (0,len(new_min_index)):
        print(str(k+1)+'.'+str(Title[new_min_index][k])+ ' | Genres = ' +str(Gn[new_min_index][k]) + ' | Keywords = ' + str(Keys[new_min_index][k]) +' | Director = ' + str(Director[new_min_index][k]) + ' | Lead Cast = ' + str(Cast[new_min_index][k][0]) + ' | Ratings = ' + str(Rating[new_min_index][k]))
    des = str(input("Are you done? Enter 'yes' to stop / 'no' to continue:"))
    if des=='yes':
       print('Thank you!')
