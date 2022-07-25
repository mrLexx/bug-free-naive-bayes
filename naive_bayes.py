import re
from collections import defaultdict
from pandas import DataFrame


class NaiveBayesGaussian:
    """
    Naive Bayes Normal Distribution Classifire
    """
    pi = 3.1415926535
    e = 2.71828182845

    def __init__(self):
        pass

    @staticmethod
    def mean(var):
        """
        Mean
        :param var:
        :return:
        """
        m = 0
        len(var)
        for i in var:
            m += i
        m = m / len(var)
        return round(m, 1)

    @staticmethod
    def st_dev(var):
        """
        Standard Deviation
        :param var:
        :return:
        """
        q = 0
        m = NaiveBayesGaussian.mean(var)

        for x in var:
            q += (x - m) ** 2
        q = (q / (len(var) - 1)) ** 0.5

        return round(q, 1)

    @staticmethod
    def posterior_probability(x, data):
        """
        Probability density function, gaussian
        :param x:
        :param data:
        :return:
        """
        m = NaiveBayesGaussian.mean(data)
        q = NaiveBayesGaussian.st_dev(data)

        p = (1 / ((2 * NaiveBayesGaussian.pi) ** 0.5 * q))
        p *= NaiveBayesGaussian.e ** (-1 * (((x - m) ** 2) / (2 * (q ** 2))))

        return round(p, 5)

    @staticmethod
    def probability(x, main_data, opponent_data):
        """
        Compare probability between two
        :param x:
        :param main_data:
        :param opponent_data:
        :return:
        """

        p_main = NaiveBayesGaussian.posterior_probability(x, data=main_data)
        p_opponent = NaiveBayesGaussian.posterior_probability(x, data=opponent_data)
        return round(((p_main * 100) / (p_main + p_opponent))) / 100
        # return round(p, 5)


class NaiveBayesMultinomial:
    """
    Naive Bayes Text Classifire
    """

    def __init__(self):
        self.show_debug = False

        self.index_class1 = 0
        self.index_class2 = 1
        self.stem_func = None
        self.stem_enable = True
        self.stem_lang = 'en'
        self.class_title = ['Class 01', 'Class 02']
        self._stop_words = None

        self.count_all = -1
        self.count_class1 = -1
        self.count_class2 = -1

        self.words_all = defaultdict(lambda: [0, 0])
        self.words_without_stop = defaultdict(lambda: [0, 0])

        pass

    def debug_print(self, text='', end='\n'):
        if self.show_debug:
            print(text, end=end)

    def load_stop_words(self, file_path):
        file = open(file_path, 'r')
        self._stop_words = []
        for line in file.readlines():
            self._stop_words.append(line.strip())

    def stem(self, word):
        if self.stem_enable and self.stem_func is not None:
            word = self.stem_func(word, self.stem_lang)
        return word

    def get_all_words(self, text):
        text = text.lower()
        all_words = []
        for i in re.findall("[a-zа-я]+", text):
            all_words.append(self.stem(i))
        return all_words

    @staticmethod
    def probability_text(count_ratio, pr_est, count_p, count_all):
        pr = 1
        for i in pr_est:
            pr *= (i + 1) / (count_p + count_all)
        pr *= count_ratio
        return pr

    def calculate_counters(self, df: DataFrame = None, column='class1', conditions=None):
        if conditions is None:
            conditions = [1, 0]

        if df is not None and column is not None:
            self.count_all = df[column].count()
            self.count_class1 = df[column].loc[df[column] == conditions[self.index_class1]].count()
            self.count_class2 = df[column].loc[df[column] == conditions[self.index_class2]].count()

        return [
            self.count_all,
            self.count_class1,
            self.count_class2
        ]

    def get_counters(self):
        return [
            self.count_all,
            self.count_class1,
            self.count_class2
        ]

    def probability(self, text: str, mode=None, words=None):
        if mode is None:
            mode = self.index_class1

        if words is None:
            words = self.words_without_stop

        pr_est_class1 = []
        pr_est_class2 = []
        for word in self.get_all_words(text):
            if word in words:
                pr_est_class1.append(words[word][self.index_class1])
                pr_est_class2.append(words[word][self.index_class2])
        prob_class1 = self.probability_text(
            self.count_class1 / self.count_all,
            pr_est_class1,
            self.count_class1,
            len(words)
        )
        prob_class2 = self.probability_text(
            self.count_class2 / self.count_all,
            pr_est_class2,
            self.count_class2,
            len(words)
        )
        # print(prob_class1, prob_class2)
        if mode == self.index_class1:
            return round(((prob_class1 * 100) / (prob_class1 + prob_class2))) / 100
        else:
            return round(((prob_class2 * 100) / (prob_class1 + prob_class2))) / 100

    def test_classifier(self, df_check: DataFrame, words=None, debug: bool = None):
        # global class_title, index_class1, index_class2
        if words is None:
            words = self.words_without_stop

        if debug == True or debug == False:
            self.show_debug = debug

        # count_class1, count_class2, count_all = counter

        cnt_good, cnt_bad = [0, 0], [0, 0]
        # words_work = words_all
        for index, row in df_check.iterrows():
            text = row['text']
            # self.debug_print(text)

            pr_est_class1 = []
            pr_est_class2 = []

            for word in self.get_all_words(text):
                if word in words:
                    pr_est_class1.append(words[word][self.index_class1])
                    pr_est_class2.append(words[word][self.index_class2])

            prob_class1 = self.probability_text(
                self.count_class1 / self.count_all,
                pr_est_class1,
                self.count_class1,
                len(words)
            )
            prob_class2 = self.probability_text(
                self.count_class2 / self.count_all,
                pr_est_class2,
                self.count_class2,
                len(words)
            )

            self.debug_print(f'Probability {self.class_title[self.index_class1]}: {prob_class1}')
            self.debug_print(f'Probability {self.class_title[self.index_class2]}: {prob_class2}')

            if prob_class1 > prob_class2:
                self.debug_print('Calc: {0} {1:.2f}%'.format(self.class_title[self.index_class1],
                                                             prob_class1 * 100 / (prob_class2 + prob_class1)))
                if row['class1'] == 1:
                    self.debug_print(f'Sample: {self.class_title[self.index_class1]}, ', end='')
                    self.debug_print('Good')
                    cnt_good[self.index_class1] += 1
                else:
                    self.debug_print(f'Sample: {self.class_title[self.index_class2]}, ', end='')
                    self.debug_print('Bad')
                    cnt_bad[self.index_class1] += 1

            else:
                self.debug_print(
                    'Calc: {0} {1:.2f}%'.format(self.class_title[self.index_class2],
                                                prob_class2 * 100 / (prob_class2 + prob_class1)))
                if row['class1'] == 1:
                    self.debug_print(f'Sample: {self.class_title[self.index_class1]}, ', end='')
                    self.debug_print('Bad')
                    cnt_bad[self.index_class2] += 1
                else:
                    self.debug_print(f'Sample: {self.class_title[self.index_class2]}, ', end='')
                    self.debug_print('Good')
                    cnt_good[self.index_class2] += 1
            # self.debug_print()
        return cnt_good, cnt_bad

    def print_result(self, cnt_good, cnt_bad):
        # global class_title, index_class1, index_class2
        print('Результат:')
        print('Тип {0}: попаданий ({1}), ошибок({2})'.format(
            self.class_title[self.index_class1],
            cnt_good[self.index_class1],
            cnt_bad[self.index_class1]))
        print('Тип {0}: попаданий ({1}), ошибок({2})'.format(
            self.class_title[self.index_class2],
            cnt_good[self.index_class2],
            cnt_bad[self.index_class2]))
        print('Всего, попаданий: {2}%'.format(
            cnt_good[self.index_class1] + cnt_good[self.index_class2],
            cnt_bad[self.index_class1] + cnt_bad[self.index_class2],
            round(((cnt_good[self.index_class1] + cnt_good[self.index_class2]) * 100) / (
                    (cnt_good[self.index_class1] + cnt_good[self.index_class2]) + (
                    cnt_bad[self.index_class1] + cnt_bad[self.index_class2])), 2))
        )
        print()

    def calculate_words(self, df: DataFrame):
        """
        Deprecated, user train
        :param df:
        :return:
        """
        return self.train(df=df)

    def train(self, df: DataFrame):
        """
        :param df:
        :return:
        """
        # global index_class1, index_class2
        self.words_all = defaultdict(lambda: [0, 0])
        self.words_without_stop = defaultdict(lambda: [0, 0])

        for index, row in df.iterrows():
            for word in self.get_all_words(row['text']):
                # if stem_func is not None:
                #     word = stem_func(word)
                if row['class1'] == 1:
                    i = self.index_class1
                else:
                    i = self.index_class2
                self.words_all[word][i] += 1

                if word not in self._stop_words:
                    self.words_without_stop[word][i] += 1

        return self.get_model()

    def get_model(self):
        return [self.words_all, self.words_without_stop]

    def set_title(self, class1_title, class2_title):
        self.class_title[self.index_class1] = class1_title
        self.class_title[self.index_class2] = class2_title


class NaiveBayesCategorical:
    def __init__(self):
        self.likelihood_table = {}
        self.class_column = ''
        pass

    def get_likelihood_table(self):
        return self.get_model()

    def get_model(self):
        """
        Get likelihood table
        :return:
        """
        return self.likelihood_table

    def probability(self, data_check: dict):
        p = {
            'answer': self.class_column,
            'probability': {}
        }
        for cl in self.likelihood_table:
            tmp_p = 1
            for param in data_check:
                cnt = self.likelihood_table[cl][param]['_data_'].get(data_check[param])
                cnt_all = self.likelihood_table[cl][param]['_cnt_']
                if cnt is None:
                    cnt = 1
                    cnt_all += 1
                tmp_p *= (cnt / cnt_all)
            p['probability'][cl] = tmp_p
        return p

    @staticmethod
    def prepare_for_query(val):
        """
        Prepare value for DataFrame.query: check str or other type
        :param val:
        :return:
        """
        if isinstance(val, str):
            return f'"{val}"'
        else:
            return f'{val}'

    def train(self, df: DataFrame, class_column: str):
        self.class_column = class_column
        class_values = df[class_column].unique()
        p = {}
        for i in class_values:
            p[i] = 0
        self.likelihood_table = {}
        for cl in class_values:
            self.likelihood_table[cl] = {}
            for column in df.head(0):
                if column != class_column:
                    query = f'{class_column}=={self.prepare_for_query(cl)}'
                    cnt_all = len(df.query(query)[column])
                    self.likelihood_table[cl][column] = {
                        '_cnt_': cnt_all,
                        '_data_': {},
                    }
                    for val in (df[column].unique()):
                        query = f'{column}=={self.prepare_for_query(val)} & {class_column}=={self.prepare_for_query(cl)}'
                        cnt = len(df.query(query)[column])
                        self.likelihood_table[cl][column]['_data_'][val] = cnt
