import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')


def getType(content, loaded_model):
  content = re.sub('[^a-zA-Z]', ' ', content)
  content = content.lower()
  content = content.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  content = [ps.stem(word) for word in content if not word in set(all_stopwords)]
  content = ' '.join(content)
  content = [content]
  cv = CountVectorizer(max_features = 1500)
  new_X_test = cv.transform(content).toarray()
  new_y_pred = loaded_model.predict(new_X_test)
  return new_y_pred

if __name__=="__main__":
  loaded_model = pickle.load(open("model.pkl", "rb"))
  output = getType(str(input("Enter content...")), loaded_model)
  if output[0]==1:
    print("Novel")
  else :
    print("Poem")
