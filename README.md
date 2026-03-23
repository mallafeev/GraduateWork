# Phylo App Prototype

Что уже умеет прототип:
- загружать alignment из `.nex/.nexus/.txt` и `.fasta`;
- извлекать матрицу из TreeBASE-подобного NEXUS;
- строить филогенетическое дерево методом Neighbor Joining;
- показывать ASCII-представление дерева;
- выводить таблицу внутренних узлов, глубины и длины ветвей;
- сохранять построенное дерево в формате Newick.

## Структура
- `main.py` — точка входа
- `app_state.py` — состояние приложения
- `parsers/nexus_parser.py` — парсинг NEXUS/FASTA
- `phylo/distance.py` — p-distance с pairwise deletion
- `phylo/nj_builder.py` — построение NJ-дерева
- `phylo/tree_utils.py` — сводка и работа с узлами дерева
- `services/analysis_service.py` — связывает парсер и дерево
- `ui/main_window.py` — графический интерфейс
