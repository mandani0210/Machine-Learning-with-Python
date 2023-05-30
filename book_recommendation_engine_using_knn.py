import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
!unzip book-crossings.zip

books_filename   = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

df_ratings.head()
df_books.isnull().sum()
df_ratings.isnull().sum()
df_books.dropna(inplace=True)
df_books.isnull().sum()
df_ratings.shape

ratings = df_ratings['user'].value_counts()
ratings.sort_values(ascending=False).head()

len(ratings[ratings < 200])
df_ratings['user'].isin(ratings[ratings < 200].index).sum()

df_ratings_rm = df_ratings[
 ~df_ratings['user'].isin(ratings[ratings < 200].index)
]
df_ratings_rm.shape

ratings = df_ratings['isbn'].value_counts()
ratings.sort_values(ascending=False).head()


len(ratings[ratings < 100])
df_books['isbn'].isin(ratings[ratings < 100].index).sum()

df_ratings_rm = df_ratings_rm[
  ~df_ratings_rm['isbn'].isin(ratings[ratings < 100].index)
]
df_ratings_rm.shape

books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True"]

for book in books:
  print(df_ratings_rm.isbn.isin(df_books[df_books.title == book].isbn).sum())
  
df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T

df.index = df.join(df_books.set_index('isbn'))['title']

df = df.sort_index()
df.head()

df.loc["The Queen of the Damned (Vampire Chronicles (Paperback))"][:5]

model = NearestNeighbors(metric='cosine')
model.fit(df.values)

NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                 metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                 radius=1.0)

df.iloc[0].shape

title = 'The Queen of the Damned (Vampire Chronicles (Paperback))'
df.loc[title].shape

distance, indice = model.kneighbors([df.loc[title].values], n_neighbors=6)
print(distance)
print(indice)

df.iloc[indice[0]].index.values

pd.DataFrame({
    'title'   : df.iloc[indice[0]].index.values,
    'distance': distance[0]
}) \
.sort_values(by='distance', ascending=False)

# function to return recommended books - this will be tested
def get_recommends(title = ""):
  try:
    book = df.loc[title]
  except KeyError as e:
    print('The given book', e, 'does not exist')
    return

  distance, indice = model.kneighbors([book.values], n_neighbors=6)

  recommended_books = pd.DataFrame({
      'title'   : df.iloc[indice[0]].index.values,
      'distance': distance[0]
    }) \
    .sort_values(by='distance', ascending=False) \
    .head(5).values

  return [title, recommended_books]

get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")



