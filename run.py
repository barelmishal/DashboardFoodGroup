from posixpath import split
from xml.etree.ElementInclude import include
import pandas as pd 
import numpy as np 

from helpers import genral_functions as gf
from helpers.constants import *




def main():
  # created labled csv 
  # gf.create_csv_with_lables_kmeans( ISRAEL_OFFICIAL_DATA_PATH, MACROS, NUMBER_OF_FOOD_SUB_GROUPS, 'latin-1', '_labled_csv')
  # gf.create_csv_with_lables_AffinityPropagation( ISRAEL_OFFICIAL_DATA_PATH, MACROS, NUMBER_OF_FOOD_SUB_GROUPS, 'latin-1', 'AffinityPropagation_labled')
  # gf.create_df_labled_with_names_ids(ISRAEL_OFFICIAL_DATA_PATH, LABELED_FOODS_FOR_SWITCH_NUMBERS_TO_NAMES)
  # words = pd.Series(gf.count_all_words(df['shmmitzrach'])).reset_index()
  
  df = gf.make_dataframe(ISRAEL_OFFICIAL_DATA_PATH)

  mask = gf.mask_by_part_of_a_word_in_df(df['shmmitzrach'], FILTER, include = False)
  df = df[mask]
  

  return

  
if __name__ == '__main__':
  # קוד לסיווג קבוצות מזון בעזרת SKLEARN 

  

  main()




  
