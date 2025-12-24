import numpy as np
import keras as kr
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загружаем датасет
dataset = pd.read_csv("comparison_ai_and_human.csv")

# Перемешиваем данные
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Признаки (5 числовых)
feature_cols = ['length_chars', 'length_words', 'quality_score', 'sentiment', 'plagiarism_score']
X = dataset[feature_cols].astype(np.float32).to_numpy()

# Кодируем label
dataset['label_code'] = pd.factorize(dataset['label'])[0]
Y = dataset['label_code'].to_numpy().reshape(-1, 1).astype(np.float32)
Y = kr.utils.to_categorical(Y, 2)

# Разбиваем на train/val/test (80/10/10)
X_all_train, X_test, Y_all_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_all_train, Y_all_train, random_state=0, test_size=0.1)

# Создаем модель
model = Sequential()
model.add(Dense(8, input_shape=(5,), kernel_initializer='random_uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

# Компилируем
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем ТОЧНО 100 эпох
history = model.fit(X_train, Y_train, batch_size=50, epochs=100, verbose=1,
                    shuffle=True, validation_data=(X_val, Y_val))

print(history.history.keys())

# ✅ УЛУЧШЕННЫЕ ГРАФИКИ в стиле профессионального шаблона
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ГРАФИК 1: ТОЧНОСТЬ (Accuracy)
axes[0].plot(history.history['accuracy'],
            label='Тренировочная точность',
            color='blue',
            linewidth=2.5,
            alpha=0.8)

axes[0].plot(history.history['val_accuracy'],
            label='Валидационная точность',
            color='orange',
            linewidth=2.5,
            alpha=0.8)

# Добавляем области для лучшей визуализации
axes[0].fill_between(range(len(history.history['accuracy'])),
                    history.history['accuracy'],
                    alpha=0.2,
                    color='blue')

axes[0].fill_between(range(len(history.history['val_accuracy'])),
                    history.history['val_accuracy'],
                    alpha=0.2,
                    color='orange')

axes[0].set_title('Динамика точности обучения',
                 fontsize=16,
                 fontweight='bold',
                 pad=20)

axes[0].set_xlabel('Эпоха', fontsize=12)
axes[0].set_ylabel('Точность', fontsize=12)
axes[0].legend(loc='lower right', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xlim([0, len(history.history['accuracy'])])
axes[0].set_ylim([0.4, 1.0])

# Добавляем аннотацию с финальной точностью
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
axes[0].annotate(f'Финальная: {final_train_acc:.3f}',
                xy=(len(history.history['accuracy']) - 1, final_train_acc),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color='blue')

axes[0].annotate(f'Финальная: {final_val_acc:.3f}',
                xy=(len(history.history['val_accuracy']) - 1, final_val_acc),
                xytext=(10, -20),
                textcoords='offset points',
                fontsize=10,
                color='orange')

# ГРАФИК 2: ПОТЕРИ (Loss)
axes[1].plot(history.history['loss'],
            label='Тренировочные потери',
            color='green',
            linewidth=2.5,
            alpha=0.8)

axes[1].plot(history.history['val_loss'],
            label='Валидационные потери',
            color='red',
            linewidth=2.5,
            alpha=0.8)

# Добавляем области
axes[1].fill_between(range(len(history.history['loss'])),
                    history.history['loss'],
                    alpha=0.2,
color='green')

axes[1].fill_between(range(len(history.history['val_loss'])),
                    history.history['val_loss'],
                    alpha=0.2,
                    color='red')

axes[1].set_title('Динамика потерь обучения',
                 fontsize=16,
                 fontweight='bold',
                 pad=20)

axes[1].set_xlabel('Эпоха', fontsize=12)
axes[1].set_ylabel('Потери', fontsize=12)
axes[1].legend(loc='upper right', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xlim([0, len(history.history['loss'])])

# Динамический расчет ylim для потерь
max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
axes[1].set_ylim([0, max_loss * 1.1])

# Добавляем аннотацию с финальными потерями
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
axes[1].annotate(f'Финальная: {final_train_loss:.3f}',
                xy=(len(history.history['loss']) - 1, final_train_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color='green')

axes[1].annotate(f'Финальная: {final_val_loss:.3f}',
                xy=(len(history.history['val_loss']) - 1, final_val_loss),
                xytext=(10, -20),
                textcoords='offset points',
                fontsize=10,
                color='red')

# Настройка общего вида
plt.suptitle('Результаты обучения нейронной сети для классификации AI vs Human',
            fontsize=18,
            fontweight='bold',
            y=1.02)

plt.tight_layout()
plt.show()

# Оценка
res_test = model.evaluate(X_test, Y_test, batch_size=1, verbose=1)
print('\n# Оцениваем на тестовых данных')
print("Test Loss: ", res_test[0])
print("Test Accuracy: %.2f%%" % (res_test[1]*100))

# Предсказания (первые 20)
probs = model.predict(X_test)
pred_class = [np.argmax(prob) for prob in probs]
Y_real = [np.argmax(y) for y in Y_test]

labels_class = ['human', 'ai']
print("\nПервые 20 предсказаний:")
for i in range(min(20, len(pred_class))):
    print('Predicted: ', "true " if pred_class[i] == Y_real[i] else "false ",
          labels_class[pred_class[i]], ' Real class:', labels_class[Y_real[i]])
