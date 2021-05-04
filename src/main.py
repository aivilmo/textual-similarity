from evaluator import Evaluator
from preprocessor import Preprocessor
from similarity import Similarity
from plot import Plot
from evaluator import Evaluator
import argparse
from statistics import mean


class Main:
    def __init__(self):
        self.m_preprocessor = Preprocessor()
        self.m_similarity = Similarity()
        self.m_plt = Plot()
        self.m_evaluator = Evaluator()
        self.m_file = "sts-train"
        self.m_metric = "path"
        self.m_ic = "brown"
        self.m_metric_w2v = "cosine"
        self.m_metric_t = "path"
        self.m_thr = 20
        self.m_mode = "ontology"

    def arg_parse(self):
        parser = argparse.ArgumentParser(
            description="Calculus of similarity", add_help=True
        )

        parser.add_argument(
            "-f",
            dest="file",
            action="store",
            type=str,
            choices=["sts-dev", "sts-test", "sts-train"],
            help="Select the csv file to calculate similarity.",
        )

        parser.add_argument(
            "-m",
            dest="metric",
            action="store",
            type=str,
            choices=["path", "lch", "wup", "res", "jcn", "lin"],
            help="Select the metric to calculate similarity.",
        )

        parser.add_argument(
            "-i",
            dest="ic",
            action="store",
            type=str,
            choices=["brown", "treebank"],
            help="Select the Information Content corpus.",
        )

        parser.add_argument(
            "-mw",
            dest="metric_w2v",
            action="store",
            type=str,
            choices=["cosine", "euclidean", "dot"],
            help="Select the metric to calculate similarity word2vec.",
        )

        parser.add_argument(
            "-mt",
            dest="metric_t",
            action="store",
            type=str,
            choices=["path", "jcn", "euclidean"],
            help="Select the metric to calculate textual similarity.",
        )

        parser.add_argument(
            "-t",
            dest="thr",
            action="store",
            type=int,
            choices=[20, 30, 40],
            help="Select the threshold to calculate textual similarity.",
        )

        args = parser.parse_args()
        if args.file != None:
            self.m_file = args.file
        if args.metric != None:
            self.m_mode = "ontology"
            self.m_metric = args.metric
        if args.ic != None:
            self.m_ic = args.ic
        if args.metric_w2v != None:
            self.m_mode = "word2vec"
            self.m_metric_w2v = args.metric_w2v
        if args.metric_t != None:
            self.m_mode = "textual"
            self.m_metric_t = args.metric_t
        if args.thr != None:
            self.m_thr = args.thr

        self.m_data_frame = self.m_preprocessor.initialize(
            "..\\stsbenchmark\\" + self.m_file + ".csv"
        )

    def main(self):
        df_original = self.m_data_frame.copy()
        stats_original = self.data_frame_stats(self.m_data_frame)
        print("Printing original stats")
        print(stats_original)

        df_filtered = self.m_preprocessor.tokenize_and_filter_dataset(self.m_data_frame)
        print(
            "Calcul similarity by "
            + self.m_mode
            + " similarity, using "
            + self.m_metric_t
        )

        if self.m_mode == "ontology":
            self.m_data_frame = self.m_similarity.calculate_ontology_similarity(
                self.m_metric, self.m_ic, df_filtered
            )
        elif self.m_mode == "word2vec":
            self.m_data_frame = self.m_similarity.calculate_word2vec_similarity(
                self.m_metric_w2v, df_filtered
            )
        elif self.m_mode == "textual":
            idf = self.m_preprocessor.idf_from_dataset(self.m_data_frame)
            self.m_data_frame = self.m_similarity.calculate_textual_similarity(
                self.m_thr, self.m_metric_t, df_filtered, idf
            )
        df_syntetic = self.m_data_frame.copy()
        stats_syntetic = self.data_frame_stats(self.m_data_frame)
        stats_total = stats_original.join(stats_syntetic, lsuffix="_o", rsuffix="_s")
        print("Printing all stats")
        print(stats_total)
        #self.m_plt.plot_scatter(stats_total)
        self.m_evaluator.evaluate(df_original, df_syntetic)
        

    def plot_base_data(self):
        for row in self.m_data_frame.itertuples():
            self.m_plt.prepare_data_to_plot(row.set, row.similarity)
        self.m_plt.sub_plot_bar(self.m_file)

    def plot_pos_tagged_data(self):
        df = self.m_preprocessor.tag_dataset(self.m_data_frame)
        for row in df.itertuples():
            self.m_plt.prepare_tags_to_plot(row.set, row.sentence_a, row.sentence_b)
        self.m_plt.sub_plot_bar(self.m_file)

    def calculate_sentences_length(self):
        df = self.m_preprocessor.tag_dataset(self.m_data_frame)
        lengths = {}
        for row in df.itertuples():
            if lengths.get(row.set, []) == []:
                lengths[row.set] = [len(row.sentence_a), len(row.sentence_b)]
                continue
            lengths[row.set].append(len(row.sentence_a))
            lengths[row.set].append(len(row.sentence_b))
        for key in lengths.keys():
            lengths[key] = mean(lengths[key])
        print(lengths)

    def data_frame_stats(self, df):
        return df.groupby(["set"])["similarity"].agg(["mean", "std", "median"])


if __name__ == "__main__":
    main = Main()
    main.arg_parse()
    main.main()
