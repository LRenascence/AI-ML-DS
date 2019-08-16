import pandas as pd
import numpy as np

r_cols = ['userid', 'movieid', 'rating']
ratings = pd.read_csv('ml-100k/u.data', encoding = "ISO-8859-1", sep = '\\t', names = r_cols, usecols = range(3))

m_cols = ['movieid', 'title']
movies = pd.read_csv('ml-100k/u.item', encoding = "ISO-8859-1", sep = '|', names = m_cols, usecols = range(2))

# merge ratings and movies like this
# movieid             title  userid  rating
# 1  Toy Story (1995)     308       4
# 1  Toy Story (1995)     287       5
# 1  Toy Story (1995)     148       4
# 1  Toy Story (1995)     280       4
# 1  Toy Story (1995)      66       3
ratings = pd.merge(movies, ratings)

# convert table using pivot_table like this
#title   'Til There Was You (1997)  ...  Á köldum klaka (Cold Fever) (1994)
# userid                             ...
# 1                             NaN  ...                                 NaN
# 2                             NaN  ...                                 NaN
# 3                             NaN  ...                                 NaN
# 4                             NaN  ...                                 NaN
# 5                             NaN  ...                                 NaN
# #[5 rows x 1664 columns]
movieRatings = ratings.pivot_table(index = ['userid'], columns = ['title'], values = 'rating')

# extract the movie rating info
starWarsRatings = movieRatings['Star Wars (1977)']
# get the similarity
similarMovies = movieRatings.corrwith(starWarsRatings)
# drop the NaN values
similarMovies = similarMovies.dropna()
# sort the similarity score
similarMovies = similarMovies.sort_values(ascending = False)


# some obscure movies show up
# improving similarity quality
# get the number of people watch this movie
movieStats = ratings.groupby('title').agg({'rating':[np.size, np.mean]})
# keep the movie that be watched by 100 people and more
popularMovies = movieStats['rating']['size'] >= 100
# combine the popular movies and similar movies
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns = ['similarity']))
# get the top 15 similar movies
df = df.sort_values(['similarity'], ascending = False)[:15]
print(df)