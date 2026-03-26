# Phylo App Prototype v4

Что добавлено:
- загрузка `model_v1.pkl`
- предсказание bootstrap через Random Forest для внутренних узлов
- отображение predicted bootstrap в таблице
- подсветка ветвей дерева по predicted bootstrap
- более аккуратная визуализация, подписи таксонов справа

## Зависимости

```bash
python -m pip install pyqt5 biopython matplotlib pandas joblib scikit-learn numpy
```

## Запуск

```bash
python main.py
```

## Как использовать

1. Загрузить alignment (`.nex/.nexus/.fasta`)
2. Построить NJ-дерево или загрузить готовое `.nwk`
3. Загрузить `model_v1.pkl`
4. Нажать **Предсказать bootstrap**
5. Смотреть результаты в таблице и на дереве
