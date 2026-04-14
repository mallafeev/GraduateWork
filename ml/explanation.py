from __future__ import annotations

import pandas as pd


FEATURE_LABELS = {
    'branch_length': 'длина ветви',
    'depth': 'глубина узла',
    'n_leaves_subtree': 'число листьев в поддереве',
    'subtree_fraction': 'доля поддерева',
    'subtree_balance': 'баланс поддерева',
    'mean_child_branch_length': 'средняя длина дочерних ветвей',
    'std_child_branch_length': 'разброс длин дочерних ветвей',
    'taxa_count': 'число таксонов',
    'alignment_length': 'длина выравнивания',
    'gap_fraction_global': 'глобальная доля пропусков',
    'gc_mean_global': 'средний GC по всему выравниванию',
    'gc_std_global': 'разброс GC по всему выравниванию',
    'variable_site_fraction_global': 'доля вариабельных сайтов',
    'gap_fraction_clade': 'доля пропусков в кладе',
    'gc_mean_clade': 'средний GC в кладе',
    'gc_std_clade': 'разброс GC в кладе',
    'mean_pairwise_pdist_clade': 'средняя p-distance внутри клады',
}


class FeatureExplanationBuilder:
    def build(self, X: pd.DataFrame, feature_importance_map: dict[str, float]) -> list[dict]:
        if X.empty:
            return []
        numeric = X.astype(float)
        medians = numeric.median(axis=0)
        stds = numeric.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        importances = {col: float(feature_importance_map.get(col, 0.0)) for col in numeric.columns}
        if not any(importances.values()):
            importances = {col: 1.0 / max(1, len(numeric.columns)) for col in numeric.columns}
        results = []
        for _, row in numeric.iterrows():
            items = []
            for col in numeric.columns:
                value = float(row[col]) if pd.notna(row[col]) else 0.0
                score = abs((value - float(medians[col])) / float(stds[col])) * importances.get(col, 0.0)
                direction = 'выше среднего' if value >= float(medians[col]) else 'ниже среднего'
                items.append({
                    'feature': col,
                    'label': FEATURE_LABELS.get(col, col),
                    'value': value,
                    'importance': float(importances.get(col, 0.0)),
                    'score': float(score),
                    'direction': direction,
                })
            top = sorted(items, key=lambda x: x['score'], reverse=True)[:5]
            text_lines = ['Параметры, сильнее всего влияющие на predicted bootstrap:']
            for item in top:
                text_lines.append(
                    f"• {item['label']}: {item['value']:.4f} ({item['direction']}, важность модели {item['importance']:.3f})"
                )
            results.append({'items': top, 'text': '\n'.join(text_lines)})
        return results
