from nltk.corpus import wordnet, wordnet_ic
from statistics import mean
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from gensim.models import KeyedVectors
import gensim.downloader
import numpy


class Similarity:
    """docstring for Similitary."""

    MAX_VALUE = 1
    THRESHOLD = {
        20: {"path": 0.80, "jcn": "7.5e+299", "euclidean": 5.667423248291016},
        30: {
            "path": 0.70,
            "jcn": "6.666666666666667e+299",
            "euclidean": 5.49223530292511,
        },
        40: {"path": 0.60, "jcn": "6e+299", "euclidean": 5.349452575047811},
    }

    def calculate_ontology_similarity(self, metric, ic, df):
        ic = wordnet_ic.ic("ic-" + ic + ".dat")
        df["similarity"] = df.apply(
            lambda x: self.ontology_similarity(metric, ic, x.sentence_a, x.sentence_b),
            axis=1,
        )
        #norm = df.sort_values(by=["similarity"])
        return self.normalize_df(df)

    def calculate_word2vec_similarity(self, metric, df):
        model = KeyedVectors.load("model.kv")
        df["similarity"] = df.apply(
            lambda x: self.word2vec_similarity(
                metric, model, x.sentence_a, x.sentence_b
            ),
            axis=1,
        )
        #norm = df.sort_values(by=["similarity"])
        return self.normalize_df(df)

    def calculate_textual_similarity(self, thr, metric, df, idf):
        model, ic = None, None
        if metric == "euclidean":
            model = KeyedVectors.load("model.kv")
        if metric == "jcn":
            ic = wordnet_ic.ic("ic-brown.dat")
        print("Calculating textual similarity from metric " + metric)
        df["similarity"] = df.apply(
            lambda x: self.textual_similarity(
                thr, metric, ic, model, x.sentence_a, x.sentence_b, idf
            ),
            axis=1,
        )
        return self.normalize_df(df)

    def ontology_similarity(self, metric, ic, tokens_a, tokens_b):
        sims = []
        for token_a in tokens_a:
            syns1 = wordnet.synsets(token_a[0], token_a[1])
            sims.append(
                max(
                    [
                        self.synsets_similarity(
                            metric, ic, syns1, wordnet.synsets(token_b[0], token_b[1])
                        )
                        for token_b in tokens_b
                    ],
                    default=0,
                )
            )
        return mean(sims) if sims != [] else 0

    def word2vec_similarity(self, metric, model, tokens_a, tokens_b):
        if metric == "cosine":
            return self.word2vec_cosine_similarity(tokens_a, tokens_b, model)
        elif metric == "euclidean":
            return self.word2vec_euclidean_similarity(tokens_a, tokens_b, model)
        elif metric == "dot":
            return self.word2vec_dot_similarity(tokens_a, tokens_b, model)

    def textual_similarity(self, thr, metric, ic, model, tokens_a, tokens_b, idf):
        return self.textual_overlap_similarity(
            thr, metric, ic, model, tokens_a, tokens_b
        )
        """
        return self.textual_frequency_similarity(
            thr, metric, ic, model, tokens_a, tokens_b, idf
        )
        """

    def textual_overlap_similarity(self, thr, metric, ic, model, tokens_a, tokens_b):
        overlap = 0
        thr = float(self.THRESHOLD[thr][metric])
        if metric == "euclidean":
            sim = self.word2vec_euclidean_similarity(tokens_a, tokens_b, model)
        else:
            sim = self.ontology_similarity(metric, ic, tokens_a, tokens_b)

        if float(sim) >= thr:
            overlap += 1
        return (2 * overlap) / (len(tokens_a) + len(tokens_b))

    def textual_frequency_similarity(self, thr, metric, ic, model, tokens_a, tokens_b, idf):
        sim_a, idf_a = self.local_frequency_similarity(metric, ic, model, tokens_a, tokens_b, idf)
        sim_b, idf_b = self.local_frequency_similarity(metric, ic, model, tokens_b, tokens_a, idf)
        local_sim_a = (sim_a / idf_a) if idf_a != 0 else 0
        local_sim_b = (sim_b / idf_b) if idf_b != 0 else 0
        return (local_sim_a + local_sim_b) / 2

    def local_frequency_similarity(self, metric, ic, model, tokens_a, tokens_b, idf):
        sim_a, idf_a = 0, 0
        for token_a in tokens_a:
            idf_local = idf.get(token_a[0], 0)
            if metric == "euclidean":
                sim = self.word2vec_euclidean_similarity([token_a], tokens_b, model)
            else:
                sim = self.ontology_similarity(metric, ic, [token_a], tokens_b)
            idf_a += idf_local
            sim_a += idf_local * sim
        return sim_a, idf_a

    def word2vec_euclidean_similarity(self, tokens_a, tokens_b, model):
        sims = []
        for token_a in tokens_a:
            if token_a[0] not in model.index_to_key:
                continue
            vector_a = model[token_a[0]]
            sims.append(
                max(
                    [
                        self.normalize(
                            self.euclidean_similarity(vector_a, model[token_b[0]]),
                            self.MAX_VALUE,
                        )
                        for token_b in tokens_b
                        if token_b[0] in model.index_to_key
                    ],
                    default=0,
                )
            )
        return mean(sims) if sims != [] else 0

    def word2vec_cosine_similarity(self, tokens_a, tokens_b, model):
        sims = []
        for token_a in tokens_a:
            if token_a[0] not in model.index_to_key:
                continue
            sims.append(
                max(
                    [
                        self.normalize(
                            model.similarity(token_a[0], token_b[0]),
                            self.MAX_VALUE,
                        )
                        for token_b in tokens_b
                        if token_b[0] in model.index_to_key
                    ],
                    default=0,
                )
            )
        return mean(sims) if sims != [] else 0

    def word2vec_dot_similarity(self, tokens_a, tokens_b, model):
        sims = []
        for token_a in tokens_a:
            if token_a[0] not in model.index_to_key:
                continue
            vector_a = model[token_a[0]]
            sims.append(
                max(
                    [
                        self.normalize(
                            numpy.dot(
                                numpy.array(vector_a), numpy.array(model[token_b[0]])
                            ),
                            self.MAX_VALUE,
                        )
                        for token_b in tokens_b
                        if token_b[0] in model.index_to_key
                    ],
                    default=0,
                )
            )
        return mean(sims) if sims != [] else 0

    def euclidean_similarity(self, vector_a, vector_b):
        dist = euclidean_distances([vector_a], [vector_b]).item()
        return 1 / (1 + dist)

    def synsets_similarity(self, metric, ic, syns1, syns2):
        return max(
            [
                self.synsets_similarity_by_metric(metric, ic, sense1, sense2)
                for sense1, sense2 in product(syns1, syns2)
            ],
            default=0,
        )

    def synsets_similarity_by_metric(self, metric, ic, synset_a, synset_b):
        if metric == "path":
            return self.path(synset_a, synset_b)
        elif metric == "lch":
            return self.lch(synset_a, synset_b)
        elif metric == "wup":
            return self.wup(synset_a, synset_b)
        elif metric == "res":
            return self.res(synset_a, synset_b, ic)
        elif metric == "jcn":
            return self.jcn(synset_a, synset_b, ic)
        elif metric == "lin":
            return self.lin(synset_a, synset_b, ic)

    def path(self, synset_a, synset_b):
        return self.normalize(
            self.MAX_VALUE, wordnet.path_similarity(synset_a, synset_b, verbose=True)
        )

    def lch(self, synset_a, synset_b):
        return (
            self.normalize(
                self.MAX_VALUE,
                wordnet.lch_similarity(synset_a, synset_b, verbose=True),
            )
            if synset_a.pos() == synset_b.pos()
            else 0
        )

    def wup(self, synset_a, synset_b):
        sim = wordnet.wup_similarity(synset_a, synset_b, verbose=True)
        return sim if sim != None else 0

    def res(self, synset_a, synset_b, ic):
        return (
            self.normalize(
                self.MAX_VALUE,
                wordnet.res_similarity(synset_a, synset_b, ic),
            )
            if synset_a.pos() == synset_b.pos()
            else 0
        )

    def jcn(self, synset_a, synset_b, ic):
        return (
            self.normalize(
                self.MAX_VALUE,
                wordnet.jcn_similarity(synset_a, synset_b, ic),
            )
            if synset_a.pos() == synset_b.pos()
            else 0
        )

    def lin(self, synset_a, synset_b, ic):
        if synset_a.pos() != synset_b.pos():
            return 0
        try:
            norm = self.normalize(
                self.MAX_VALUE, wordnet.lin_similarity(synset_a, synset_b, ic)
            )
            if norm == None:
                return 0
            return norm
        except ZeroDivisionError:
            return 0

    def normalize(self, max, value):
        if value == None:
            return 0
        value = float(value)
        return (value / max) * 1

    def normalize_df(self, df):
        min_max_scaler = preprocessing.MinMaxScaler()
        sims = df["similarity"].values.reshape(-1, 1)
        sims_scaled = min_max_scaler.fit_transform(sims)
        df["similarity"] = sims_scaled
        df["similarity"] = df["similarity"].apply(lambda x: x * 5)
        return df

    def load_model_w2v(self):
        model = gensim.downloader.load("word2vec-google-news-300")
        model.save("model.kv")
        return model
