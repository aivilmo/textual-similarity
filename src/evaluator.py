
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator():
    """ Class to Evaluate data, """

    def evaluate(self, df_original, df_syntetic):
        df_final = pd.DataFrame()
        for set in ["MSRpar", "MSRvid", "deft-forum", "deft-news", "headlines", "images"]:
            df_o = df_original[df_original['set'] == set]
            df_s = df_syntetic[df_syntetic['set'] == set]

            df_o.rename(columns={'similarity': 'o_'+set}, inplace = True)
            df_s.rename(columns={'similarity': set}, inplace = True)

            series_original = pd.Series(df_o['o_'+set], name='o_'+set)
            series_syntetic = pd.Series(df_s[set], name=set)

            df = pd.concat([series_original, series_syntetic], axis=1)
            df_final = df_final.append(df)

        sns.heatmap(df_final.corr(), annot=True)
        plt.show()

