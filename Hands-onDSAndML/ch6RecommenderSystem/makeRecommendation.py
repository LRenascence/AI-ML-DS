import pandas as pd


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
userRatings = ratings.pivot_table(index = ['userid'], columns = ['title'], values = 'rating')
# get the corrMatrix like this
# title                      'Til There Was You (1997)  ...  Á köldum klaka (Cold Fever) (1994)
# title                                                 ...
# 'Til There Was You (1997)                        1.0  ...                                 NaN
# 1-900 (1994)                                     NaN  ...                                 NaN
# 101 Dalmatians (1996)                           -1.0  ...                                 NaN
# 12 Angry Men (1957)                             -0.5  ...                                 NaN
# 187 (1997)                                      -0.5  ...

# corrMatrix = movieRatings.corr()
# can add some parameters to .corr() function
# min_periods the number of people that rated both movies
corrMatrix = userRatings.corr(method = 'pearson', min_periods = 100)

# get the user watch log
myRatings = userRatings.loc[1].dropna()
# recommend movies to this user
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    # retrieve similar movies to this one that the user rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # score it
    sims = sims.map(lambda x: x * myRatings[i])
    # add the score to the list
    simCandidates = simCandidates.append(sims)
# combine rows because there are duplicate movies
simCandidates = simCandidates.groupby(simCandidates.index).sum()
# sort the result
simCandidates.sort_values(inplace = True, ascending = False)
# remove the movie that the user watched
filteredCandidates = simCandidates.drop(myRatings.index, errors = 'ignore')
print(filteredCandidates.head())