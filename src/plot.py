import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plot:
    """docstring for ."""

    def __init__(self):
        self.m_lengths = {
            "sts-train": {
                "MSRpar": 1000,
                "headlines": 1999,
                "deft-news": 300,
                "MSRvid": 1000,
                "images": 1000,
                "track5.en-en": 0,
                "deft-forum": 450,
                "answers-forums": 0,
                "answer-answer": 0,
            },
            "sts-dev": {
                "MSRpar": 250,
                "headlines": 250,
                "deft-news": 0,
                "MSRvid": 250,
                "images": 250,
                "track5.en-en": 125,
                "deft-forum": 0,
                "answers-forums": 375,
                "answer-answer": 0,
            },
            "sts-test": {
                "MSRpar": 250,
                "headlines": 250,
                "deft-news": 0,
                "MSRvid": 250,
                "images": 250,
                "track5.en-en": 125,
                "deft-forum": 0,
                "answers-forums": 0,
                "answer-answer": 254,
            },
        }
        self.m_data_for_set = {}

    def sub_plot_bar(self, file):
        _, ax = plt.subplots()
        self.swap_key_value(file)
        self.do_sub_plot(ax, self.m_data_for_set, file)
        plt.show()

    def plot_scatter(self, df):
        # sns.pairplot(df, hue="set", size=4)
        # sns.scatterplot(data=df, x="set", y="mean")
        # sns.relplot(data=df, x="id", y="similarity", hue="set", col="genre", col_wrap=2)
        df = df.reset_index().melt("set", var_name="cols", value_name="vals")
        sns.catplot(x="set", y="vals", hue="cols", data=df, kind="point")
        plt.show()

    def do_sub_plot(
        self, ax, data, file, colors=None, total_width=0.8, single_width=1, legend=True
    ):
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        n_bars = len(data)
        bar_width = total_width / n_bars
        bars = []

        for i, (name, values) in enumerate(data.items()):
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
            for x, y in enumerate(values):
                bar = ax.bar(
                    x + x_offset,
                    y,
                    width=bar_width * single_width,
                    color=colors[i % len(colors)],
                )
            bars.append(bar[0])

        if legend:
            ax.legend(bars, data.keys())
        ax.set_title(file)
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Data")

    def prepare_data_to_plot(self, pair_id, similitary):
        similitary = round(float("{:.5f}".format(similitary)), 0)
        if similitary > 5:
            similitary = 5
        if self.m_data_for_set.get(pair_id, {}) == {}:
            self.m_data_for_set[pair_id] = {similitary: 1}
        else:
            if self.m_data_for_set[pair_id].get(similitary, 0) == 0:
                self.m_data_for_set[pair_id][similitary] = 1
            else:
                self.m_data_for_set[pair_id][similitary] += 1

    def prepare_tags_to_plot(self, pair_id, sentence_a, sentence_b):
        sentence_a.extend(sentence_b)
        for token in sentence_a:
            tag = self.normalize_tag(token[1])
            if self.m_data_for_set.get(pair_id, {}) == {}:
                self.m_data_for_set[pair_id] = {tag: 1}
            else:
                if self.m_data_for_set[pair_id].get(tag, 0) == 0:
                    self.m_data_for_set[pair_id][tag] = 1
                else:
                    self.m_data_for_set[pair_id][tag] += 1

    def normalize_tag(self, tag):
        if "VB" in tag:
            return "VB"
        elif "NN" in tag:
            return "NN"
        elif "JJ" in tag:
            return "JJ"
        elif "DT" in tag:
            return "DT"
        elif "IN" in tag or "TO" in tag:
            return tag
        return "OTHER"

    def swap_key_value(self, file):
        for father_key in self.m_data_for_set.keys():
            self.m_data_for_set[father_key] = {
                value / self.m_lengths[file][father_key]: key
                for key, value in sorted(self.m_data_for_set[father_key].items())
            }
        print(self.m_data_for_set)

    def calcul_percentual(self):
        total_data = self.m_data_for_set.values()
        for key, value in self.m_data_for_set.items():
            total = sum(value.keys())
            for sim in value.keys():
                num = value[sim]
                percent = num / total
                self.m_data_for_set[key][sim] = percent
        print(self.m_data_for_set)
