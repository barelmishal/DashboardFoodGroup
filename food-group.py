from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# -------------------------

import os
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
    print('labels_names - ' df)
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


# ----------------------------

from helpers import genral_functions as gf

df_kmeans = create_labeled_dataframe("kmeans")
df_minibatch_kmeans = create_labeled_dataframe("minibatch_kmeans")
df_hierarchical = create_labeled_dataframe("hierarchical")
df_dbscan = create_labeled_dataframe("dbscan")

df_kmeans_group = create_table_food_group(df_kmeans, LABELS_DIC)
df_minibatch_kmeans_group = create_table_food_group(df_minibatch_kmeans, LABELS_DIC)
df_hierarchical_group = create_table_food_group(df_hierarchical, LABELS_DIC)
df_dbscan_group = create_table_food_group(df_dbscan, LABELS_DIC)

df_words_kmeans = pd.Series(gf.count_all_words(df_kmeans['name'])).reset_index()


FOOD_GROUP_OPTIONS = [{'label': x, 'value': x} for x in range(32)]
ALGORITHMS = ["kmeans", "minibatch_kmeans", "hierarchical", "dbscan"]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Container([
    html.H1("Interactive plotly with Dash To Visulize Labling Of Food Group By Various Algorithms",
            className='mb-2', style={'textAlign':'center'}),

    dbc.Row([
        dbc.Label('Choose clustering algorithm:'),
        dbc.Col([
            dcc.Dropdown(
                id='cluster-algo',
                options=[{'label': x, 'value': x} for x in ALGORITHMS],
                value='kmeans')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label('3d scatter plot of 32 food groups:'),
            dcc.Graph(
                id='scatter-plot',
                style={'border-width': '5', 'width': '100%',
                       'height': '750px'})
        ], width=6),

        dbc.Col([
            dbc.Label('Table of food items, food groups and their macronutrients:'),
            html.Div(id='table-placeholder')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Label('Clear food groups from the plot and table:'),
        dbc.Col([
            dcc.Dropdown(
                id='food-group',
                multi=True,
                #clearable=False,
                #value=[int(dic['value']) for dic in FOOD_GROUP_OPTIONS])
                options=FOOD_GROUP_OPTIONS,
                value=[int(dic['value']) for dic in FOOD_GROUP_OPTIONS])
        ], width=12)
    ]),

    dbc.Row([
        dbc.Label('Table of food groups assigned to labels by the clustering algorithm:'),
        dbc.Col([
            html.Div(id='table-foodgroups')
        ], width=12)
    ]),

    # dbc.Row([
    #     dbc.Label('All Words In "name" column'),
    #     dbc.Col([
    #         html.Div(id='table-words')
    #     ], width=12)
    # ]),

])


# Create interactivity between components and graph
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('table-placeholder', 'children'),
    Output('table-foodgroups', 'children'),
    # Output('table-words', 'children'),
    Input('cluster-algo', 'value'),
    Input('food-group', 'value')
)
def plot_data(cluster_value, food_group_value):
    # Choose proper dataframe
    if cluster_value == "hierarchical":
        df = df_hierarchical
        df_group = df_hierarchical_group
    elif cluster_value == "kmeans":
        df = df_kmeans
        df_group = df_kmeans_group
        df_words = df_words_kmeans
    elif cluster_value == "minibatch_kmeans":
        df = df_minibatch_kmeans
        df_group = df_minibatch_kmeans_group
    else:
        df = df_dbscan
        df_group = df_dbscan_group
    df = df.astype({"label": 'category'})
    df_group = df_group.reset_index()

    # filter data based on user selection
    df_filtered = df[df.label.isin(food_group_value)]
    df_group_filtered = df_group[["index"] + food_group_value]

    # build scatter plot
    scatter_3d = px.scatter_3d(df_filtered, x='protein', y='total_fat', z='carbohydrates', size='food_energy',
                               color='label', hover_data=['name', 'label'])

    # build DataTable
    mytable = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in
                 df_filtered.loc[:, ['name', 'label', 'protein', 'total_fat', 'carbohydrates']].columns],
        data=df_filtered.to_dict('records'),
        page_size=20,
    )

    group_table = dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in [str(col) for col in df_group_filtered.columns]],
        data=df_group_filtered.to_dict('records'),
    )

    # words_table = dash_table.DataTable(
    #     id='table3',
    #     columns=[{"name": i, "id": i} for i in [str(col) for col in df_words.columns]],
    #     data=df_words.to_dict('records'),
    # )

    return scatter_3d, mytable, group_table#, words_table


if __name__ == '__main__':
    app.run_server(debug=False)