import os
from turtle import pd
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift

#from helpers.constants import LABELED_FOODS_FOR_SWITCH_NUMBERS_TO_NAMES

# AffinityPropagation, SpectralClustering too slow for our data

MACROS = ['protein', 'total_fat', 'carbohydrates']
NUMBER_OF_FOOD_GROUPS = 32
LABELS_DIC = {
    56208068:	'1.1 דגנים מלאים',
    51140668:	'1.2 דברי מאפה (לחמים) מדגן מלא',
    56204930:	'1.3 דגנים לא מלאים',
    51101009:	'1.4 דברי מאפה (לחמים) מדגן לא מלא',
    71001040:	'1.5 ירקות עמילניים',
    58126148:	'1.6 דברי מאפה מלוחים',
    53510019:	'1.7 דברי מאפה מתוקים',
    57208099:	'1.8 מידגנים (Cereals)',
    92530229:	'10.1 משקאות ממותקים בסוכר',
    64105419:	'10.2 משקאות ממותקים בתחליפי סוכר',
    92205000:	'10.3 משקאות אורז/ שיבולת שועל',
    93504000:	'10.4.1 משקאות אלכוהוליים לא מתוקים',
    93401010:	'10.4.2 משקאות אלכוהוליים שמכילים פחמימות',
    75100780:	'2 קבוצת הירקות',
    63149010:	'3 קבוצת הפירות',
    41101219:	'4 קבוצת קטניות',
    42116000:	'5.1 קבוצת אגוזים וגרעינים',
    43103339:	'5.2 קבוצת שמנים צמחיים',
    83107000:	'5.3 שומנים מהחי',
    90000007:	'6.1 מוצרי חלב לא ממותקים',
    11411609:	'6.2 מוצרי חלב ממותקים בסוכר',
    11511269:	'6.3 קבוצת מוצרי חלב ממותקים בתחליפי סוכר- כל הדיאט',
    41420010:	'6.4 קבוצת תחליפי חלב על בסיס קטניות- מוצרי סויה',
    24102010: '8.1 קבוצת הבשר- עוף הודו',
    26115108:	'8.2 קבוצת הבשר- דגים',
    23200109:	'8.3 קבוצת הבשר- בשר בקר וצאן',
    50030109:	'8.4 קבוצת הבשר- תחליפי בשר מהצומח',
    91703070:	'9. קבוצת הסוכרים',
    31104000:	'ביצים 7',
    14210079:	'מוצרי חלב דל שומן',
    95312600:	'משקה אנרגיה',
    41811939:	'תחליפי בשר (לייט)'
}


def clustering_algorithm(algorithm_name, df):
    if algorithm_name == "kmeans":
        return KMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, random_state=0).fit(df)
    elif algorithm_name == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, random_state=0, batch_size=6).fit(df)
    elif algorithm_name == "hierarchical":
        return AgglomerativeClustering(n_clusters=NUMBER_OF_FOOD_GROUPS, linkage="complete").fit(df)
    elif algorithm_name == "dbscan":
        return DBSCAN(eps=3, min_samples=2).fit(df)
    elif algorithm_name == "mean_shift":
        return MeanShift(bandwidth=2).fit(df)
    return KMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, random_state=0).fit(df)


def labels_names(df):
    labels_to_names = {}
    for key in LABELS_DIC:
        food_item = (df.loc[df['smlmitzrach'] == key])
        if food_item.iloc[0]['label'] in labels_to_names:
            labels_to_names[food_item.iloc[0]['label']] += " " + LABELS_DIC[key]
        else:
            labels_to_names[food_item.iloc[0]['label']] = LABELS_DIC[key]
    for i in range(NUMBER_OF_FOOD_GROUPS):
        if i not in labels_to_names:
            labels_to_names[i] = 'לא סווג'
        df.loc[df['label'] == i, 'labels_names'] = labels_to_names[i]
    df['labels_names'].fillna('לא סווג', inplace=True)
    return df

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))

def create_labeled_dataframe(algorithm_name):
    df_original = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'israeli_data.csv'))
    df_original = df_original.loc[:, MACROS + ['smlmitzrach', 'shmmitzrach', 'food_energy']].dropna()
    df = df_original.loc[:, MACROS]
    clusters = clustering_algorithm(algorithm_name, df)
    df['label'] = clusters.labels_
    df['smlmitzrach'] = df_original['smlmitzrach']
    df['name'] = df_original['shmmitzrach']
    df['food_energy'] = df_original['food_energy']
    df = labels_names(df)
    return df.drop(['smlmitzrach'], axis=1)


def get_matrix_food_group(unique_groups, labels_dict):
    food_group_names = labels_dict.values()
    values = {group_name: [] for group_name in food_group_names}
    for tuple_groups in unique_groups:
        for group in food_group_names:
            if group in tuple_groups[0]:
                values[group].append('True')
            else:
                values[group].append("")
    return values


def create_table_food_group(df: pd.DataFrame, labels_dict: Dict) -> pd.DataFrame:
    data_frame = df[['labels_names', 'label']]
    marged_columns = data_frame.apply(tuple, axis=1)
    unique_groups = sorted(marged_columns.unique(), key=lambda x: x[1])
    values = get_matrix_food_group(unique_groups, labels_dict)
    return pd.DataFrame.from_dict(values).T
