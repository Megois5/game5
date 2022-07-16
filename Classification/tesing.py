from pycaret.classification import *
import epitran
epi = epitran.Epitran('sin-Sinh')

Decision_Tree_Classifier = load_model('trained_models/Decision_Tree_Classifier')  # best 1
Ridge_Classifier = load_model('trained_models/Ridge_Classifier')  # best 2
SVM_Linear_Kernel = load_model('trained_models/SVM_Linear_Kernel')  # best 3

model1 = load_model('new_train_models/Ada_Boost_Classifier')
model2 = load_model('new_train_models/Gradient_Boosting_Classifier')
model3 = load_model('new_train_models/Logistic_Regression')

# input = 'ලමයෙක් අංක 1 සිට 60 දක්වා ඇති සංඛ්‍යා ලියන විට ඔහු 5 සංඛ්‍යාව කී වතාවක් ලියා තිබේද'
# input = 'ගාමිණී විදුහලේ 4056ක් ඉගෙනුම ලබයි. ඊට වඩා 396ක් වැඩියෙන් පීතර විදුහලේ සිසුන් සිටී නම් පාසල් දෙකේම සිටින මුලු සිසුන් සංඛ්‍යාව කීයද?'
# input = 'අම්මා කොකිස් 4ක් සාදන විට අක්කා කොකිස් 3ක් සාදයි. මෙලෙස දෙදෙනාම එකතු වී කොකිස් 35ක් සෑදුවේ නම් අක්කා සෑදු කොකිස් කීයද?'
# input = 'පංතියක ලමයිට දිනපතා උදේ හා හවසට බිස්කට් 4 බැගින් දෙනු ලැබේ. එම පංතියේ ලමයි 38 දෙනකු සිටී නම් දින 3ක් දීමට අවශ්‍ය වන බිස්කට් ප්‍රමානය කොපමණද'
# input = 'තාප්පයක් බැදීමට පෙදරේරුවන් 3න් දෙනෙකුට දින 10ක් ගත වෙයි. එයට පෙදරේරුවන් 5 දෙනෙකුට දින කීයක් ගත වෙයිද '
input = 'මිනිසුන් 4 දෙනෙකු දින 5ක් තුල ගොයම් කපා නිම කරයි. දින දෙකක්තුල එම වැඩය නිම කිරීමට යෙදවිය යුතු මිනිසුන් ගනන කීයද?'

input2 = epi.transliterate(input)


data = np.array([['Question'], [input]])
result = predict_model(Decision_Tree_Classifier, data=pd.DataFrame(data=data[0, 0], index=data[0:, 0], columns=data[0, 0:]))
# print('Predicted result ' + str(result))
print('old')
print(result)
print('----------------------------')


data = np.array([['Question'], [input]])
result2 = predict_model(model3, data=pd.DataFrame(data=data[0, 0], index=data[0:, 0], columns=data[0, 0:])).iat[1, 1]
# print('Predicted result ' + str(result))
print('new')
print(result2)
print(type(result2))
print('----------------------------')

