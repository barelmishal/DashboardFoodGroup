from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from clustering import create_labeled_dataframe, create_table_food_group, LABELS_DIC
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