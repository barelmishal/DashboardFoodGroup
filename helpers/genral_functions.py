import itertools
from typing import Dict
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import MiniBatchKMeans

from collections import Counter

from helpers.constants import ISRAEL_OFFICIAL_DATA_PATH, PATH

# implaments algoritems functions
def get_lables_from_clustring_AffinityPropagation_sklearn(n_d_array):
  affinity_propagation_var = AffinityPropagation(random_state=0).fit(n_d_array)
  return pd.Series(affinity_propagation_var.labels_, name='lable')

def get_lables_from_clustring_KMeans_sklearn(NUMBER_OF_FOOD_GROUP, n_d_array):
  kmeans = KMeans(n_clusters=NUMBER_OF_FOOD_GROUP, random_state=0).fit(n_d_array)
  return pd.Series(kmeans.labels_, name='lable')
  # kmeans.predict([[0, 0], [12, 3]])
  # print(kmeans.cluster_centers_)


def mask_by_part_of_a_word_in_df(string_series, pattern, include=True):
  # https://github.com/suemnjeri/Filtering-a-DataFrame/blob/master/filtering%20a%20dataframe%20by%20strings%20or%20patterns.ipynb
  '''
   include=True -> keep all foods that include the re pattren 
   include=False -> remove all foods that include the re pattren 
  '''
  mask = string_series.str.contains(pattern, case=False, na=False)
  return mask if include else ~mask


def count_all_words(series: pd.Series):
  words = itertools.chain(*[string.split() for string in series.to_numpy().flatten()])
  return Counter( words )

def get_lables_from_clustring_KMeans_sklearn(NUMBER_OF_FOOD_GROUP, n_d_array):
  # need to chack
  kmeans = MiniBatchKMeans(n_clusters=NUMBER_OF_FOOD_GROUP, random_state=0, batch_size=6).fit(n_d_array)
  return pd.Series(kmeans.labels_, name='lable')
  # kmeans.predict([[0, 0], [12, 3]])
  # print(kmeans.cluster_centers_)                        

def make_dataframe(path, encoding_for_hebrew='latin-1'):
  return pd.read_csv(path, encoding=encoding_for_hebrew)


def preper_for_labling(df, column_to_index, columns_names):
  df_indexed = df.set_index(column_to_index)
  return get_columns_as_df_without_na(df_indexed, columns_names)


def select_from_israel_offical_df_for_clustring(ISRAEL_OFFICIAL_DATA_PATH, columns_names, encoding_for_hebrew, column_to_index='smlmitzrach'):
  df = make_dataframe(ISRAEL_OFFICIAL_DATA_PATH, encoding_for_hebrew)
  return preper_for_labling(df, column_to_index, columns_names)


def get_columns_as_df_without_na(df, columns_names):
  return df.loc[:, columns_names].dropna()


def concat_labels_to_df(df, lables_as_series):
  return pd.concat([df, lables_as_series], axis=1)


def make_csv_from_df(df, file_name, encoding_for_hebrew='latin-1'):
  df.to_csv(f'csvs/{file_name}.csv', encoding=encoding_for_hebrew)


# functions that create something that combien couple of functions
def create_csv_with_lables_AffinityPropagation(path, columns_names, number_for_cluster, encoding_for_file, file_name):
  '''
  # result - (my be trying anothers parmeters and it will work? need to read more about it)
    : Affinity propagation did not converge, this model will not have any cluster centers.
    warnings.warn(

  # the reason for trying the algoritem
    @ https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
    The main drawback of Affinity Propagation is its complexity. The algorithm has a time complexity of the order , where is the number of samples and is the number of iterations until convergence. Further, the memory complexity is of the order if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation most appropriate for small to medium sized datasets.
  '''
  df_selected = select_from_israel_offical_df_for_clustring(path, columns_names, encoding_for_file)
  lables = get_lables_from_clustring_AffinityPropagation_sklearn(df_selected.values)
  concated_df = concat_labels_to_df(df_selected.reset_index(), lables)
  make_csv_from_df(concated_df, file_name)


def create_csv_with_lables_kmeans(path, columns_names, number_for_cluster, encoding_for_file, file_name):
  '''

  # the reason for trying the algoritem 
  https://scikit-learn.org/stable/modules/clustering.html#k-means
  '''
  df_selected = select_from_israel_offical_df_for_clustring(path, columns_names, encoding_for_file)
  lables = get_lables_from_clustring_KMeans_sklearn(number_for_cluster, df_selected.values)
  concated_df = concat_labels_to_df(df_selected.reset_index(), lables)
  make_csv_from_df(concated_df, file_name)


def create_nutri_records_to_wide_format(PATH):
  df = pd.read_csv(PATH)
  df = df.drop(columns=['Unnamed: 0']).dropna()
  df_wide = df[['FoodName', 'Nutrient', 'NutrientUnit', 'value']]
  df_wide = df_wide.pivot_table(columns=['Nutrient', 'NutrientUnit'], index='FoodName', values='value')
  df_wide.to_csv('csvs_from_our_records_2_samples_1_original/wide_.csv')


# function for adding to the lables names from (labled_series_indexed_foods_id_ie_smlmitzrach, labled_id_with_names_in_list)
def create_df_with_labeld_names(lables, labled_id_with_names_in_list): 
  '''
  make shour there inst a nan in (labled_series_indexed_foods_id_ie_smlmitzrach)

                lable                                     labeld_names
  smlmitzrach
  11825010        16  6.4 קבוצת תחליפי חלב על בסיס קטניות- מוצרי סויה
  56207228        22                      6.2 מוצרי חלב ממותקים בסוכר
  11000000         8                         6.1 מוצרי חלב לא ממותקים
  11111009         8                         6.1 מוצרי חלב לא ממותקים
  11111029        16  6.4 קבוצת תחליפי חלב על בסיס קטניות- מוצרי סויה
  '''
  get_2d_index = lambda i1, i2: labled_id_with_names_in_list[i1][i2]
  range_by_langth = range(len(labled_id_with_names_in_list))
  cond = [lables == lables.loc[get_2d_index(ind_name, 0)] for ind_name in range_by_langth]
  choi = [get_2d_index(ind_name, 1) for ind_name in range_by_langth]
  names_labeld = pd.Series(np.select(cond, choi, 'לא סווג'), index=lables.index, name='labeld_names')
  return concat_labels_to_df(lables, names_labeld)

# function for adding to the lables names from (labled_series_indexed_foods_id_ie_smlmitzrach, labled_id_with_names_in_list)
def create_df_labled_with_names_ids(ISRAEL_OFFICIAL_DATA_PATH, LABELED_FOODS_FOR_SWITCH_NUMBERS_TO_NAMES, set_index='smlmitzrach', MACROS=['protein', 'total_fat', 'carbohydrates']):
  df = make_dataframe(ISRAEL_OFFICIAL_DATA_PATH).set_index(set_index)[MACROS].dropna()
  labels = get_lables_from_clustring_KMeans_sklearn(32, df.values)
  labled_df = concat_labels_to_df(df.reset_index(), labels).set_index(set_index)
  labled_id_with_names_in_list = list(LABELED_FOODS_FOR_SWITCH_NUMBERS_TO_NAMES.items())
  labled_series_indexed_foods_id_ie_smlmitzrach = labled_df.lable
  return create_df_with_labeld_names(labled_series_indexed_foods_id_ie_smlmitzrach, labled_id_with_names_in_list)





