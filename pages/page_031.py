import streamlit as st
import pandas as pd

st.title("Страница 1. Модели")
st.header("1. Классификация отзывов на поликлиники")
st.divider()

st.header(" Набор данных")
nassiv_row = 70597
nassiv_col = 6
st.write(f"* Размер массива: строк`{nassiv_row}` и столбцов `{nassiv_col}`")
st.write(f"* Распределение массива по отзывам: 58% негативные и 42% позитивные")
st.write(f"* Для моделирования использованы два столбца - заголовок сообщения и само сообщение")
st.write(f"* Введено ограничение на частоту появления слов в тексте в 200 шт.")
st.write(f"* Настройка  расчета обучения моделей производилась на выборка 5%")

st.divider()
st.header(" Последовательность действий")
st.write(f"* 1. Очистка строк отзывов от неязыковых символов")
st.write(f"* 2. Токенизация строк отзывов")
st.write(f"* 3. Лемматизация слов")
st.write(f"* 4. Удаление стоп-слов")
st.write(f"* 5. Удаление одинарных букв (инициалы)")
st.write(f"* 6. Векторизация, подготовка массивов BOW, TFI-DF")
st.write(f"* 7. Введение ограничения на минимальный размер частоты появления слова в массиве отзывов")
st.write(f"* 8. Моделирование и сохранение моделей")


st.divider()
st.header(" Использованные модели")
col0, col1, col2, col3 = st.columns(spec=[0.1, 0.5, 0.2, 0.2])
with col0:
    st.write('1') 
    st.write('2') 
    st.write('3') 
    st.write('4') 
    st.write('5') 
    st.write('6') 
    st.write('7') 
    st.write('9') 
with col1:
    st.write('GaussianNB') 
    st.write('LogisticRegression') 
    st.write('SVC') 
    st.write('AdaBoostClassifier') 
    st.write('GradientBoostingClassifier') 
    st.write('LinearRegression') 
    st.write('ruBERT-tiny2') 
    st.write('RNN with attantion') 
with col2:
    st.write('BOW')    
    st.write('BOW')    
    st.write('BOW')    
    st.write('BOW')    
    st.write('BOW')    
    st.write('BOW')    
with col3:
    st.write('FTI-DF')    
    st.write('FTI-DF')    
    st.write('FTI-DF')    
    st.write('FTI-DF')    
    st.write('FTI-DF')    
    st.write('FTI-DF')    


st.divider()



epochs = 10 
train_size = 13432
val_size = 3347


st.header(" Основные параметры обучения")

st.write(f"Обычные ML-модели")
try:
    results_ML = pd.read_csv('hosp_feedback_folder/data/models_results.csv', index_col=0)
except:
    st.write("Ошибка загрузки файла результатов")
else:
    results_ML = results_ML.sort_values('Accuracy', ascending=False)
    st.table(results_ML)



st.write("ruBERT-tiny2 модель")
try:
    results_ML = pd.read_csv('hosp_feedback_folder/data/bert_model_results.csv', index_col=0)
except:
    st.write("Ошибка загрузки файла результатов")
else:
    results_ML = results_ML.sort_values('Accuracy', ascending=False)
    st.table(results_ML)

epochs = 55
st.write(f"- Число эпох: `{epochs}`")


st.write("RNN with attantion' модель")
try:
    results_ML = pd.read_csv('hosp_feedback_folder/data/RNN_model_results.csv', index_col=0)
except:
    st.write("Ошибка загрузки файла результатов")
else:
    results_ML = results_ML.sort_values('Accuracy', ascending=False)
    st.table(results_ML)

epochs = 30
st.write(f"- Число эпох: `{epochs}`")

st.divider()



