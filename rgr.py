import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_breast_cancer


# МЕДИЦИНА (ДІАБЕТ)

print("АНАЛІЗ ДАНИХ ПРО ДІАБЕТ")

url_diabetes = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

try:
    data = pd.read_csv(url_diabetes, names=columns)
    print("Побудова матриці розсіювання (зачекайте, відкриється вікно)...")
    sns.pairplot(data, hue='Outcome', diag_kind='kde')
    plt.suptitle("Матриця взаємозв'язків ознак діабету", y=1.02)
    plt.show()

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Навчання моделі Random Forest
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    rf_preds = model_rf.predict(X_test)

    print(f"\nРезультати моделі Random Forest (Діабет):")
    print(f"Точність (Accuracy): {accuracy_score(y_test, rf_preds):.4f}")
    print("\nДетальний звіт про класифікацію:")
    print(classification_report(y_test, rf_preds, target_names=['Здоровий (0)', 'Хворий (1)']))

    # Побудова матриці помилок

    print("Побудова матриці помилок...")
    cm = confusion_matrix(y_test, rf_preds)

    # Визначаємо мітки для осей
    display_labels = ['Немає діабету\n(Здоровий)', 'Є діабет\n(Хворий)']

    # Створення графічного об'єкта
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    # Візуалізація з налаштуваннями
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

    plt.title("Матриця помилок: Аналіз точності діагностики")
    plt.xlabel("Прогноз моделі (Predicted)")
    plt.ylabel("Реальний стан пацієнта (True)")
    plt.show()

except Exception as e:
    print(f"Помилка завантаження даних діабету: {e}")


# ЗАВДАННЯ 1: АНАЛОГІЧНА МОДЕЛЬ ДЛЯ РАКУ ГРУДЕЙ

print("\n ЗАВДАННЯ 1: МОДЕЛЬ ДЛЯ КЛАСИФІКАЦІЇ (Breast Cancer)")

cancer = load_breast_cancer()
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(df_cancer, y_cancer, test_size=0.2, random_state=42)

model_cancer = RandomForestClassifier(random_state=42)
model_cancer.fit(X_train_c, y_train_c)
cancer_preds = model_cancer.predict(X_test_c)

print(f"Точність на наборі Breast Cancer: {accuracy_score(y_test_c, cancer_preds):.4f}")
print("\nЗвіт для Завдання 1:")
print(classification_report(y_test_c, cancer_preds, target_names=cancer.target_names))


# ЗАВДАННЯ 2: АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК

print("\n ЗАВДАННЯ 2: АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК")

importances = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Рейтинг важливості ознак для діагностування діабету:")
print(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index, hue=importances.index, palette='magma', legend=False)
plt.title("Графік важливості медичних показників (Завдання 2)")
plt.xlabel("Коефіцієнт важливості")
plt.ylabel("Ознака")
plt.show()


# ЗАВДАННЯ 3: ПОРІВНЯННЯ МОДЕЛЕЙ

print("\n ЗАВДАННЯ 3: ПОРІВНЯННЯ МОДЕЛЕЙ (RF vs Logistic Regression) ")

model_lr = LogisticRegression(max_iter=5000)
model_lr.fit(X_train, y_train)
lr_preds = model_lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_preds)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"Точність Random Forest:      {rf_acc:.4f}")
print(f"Точність Logistic Regression: {lr_acc:.4f}")

print("\nВИСНОВОК ПОРІВНЯННЯ:")
if lr_acc > rf_acc:
    print(f"Логістична регресія показала кращий результат на {(lr_acc - rf_acc)*100:.2f}%.")
else:
    print(f"Random Forest показав кращий результат на {(rf_acc - lr_acc)*100:.2f}%.")