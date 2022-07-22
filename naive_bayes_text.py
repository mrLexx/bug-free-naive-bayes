import re
from collections import defaultdict


show_debug = False
index_class1 = 0
index_class2 = 1

class_title = ['Class 01', 'Class 02']

stem_func = None
stem_enable = True
stem_lang = 'en'


def debug_print(text='', end='\n'):
    if show_debug:
        print(text, end=end)


def load_stop_words(file_path):
    file = open(file_path, 'r')
    listWords = []

    for line in file.readlines():
        listWords.append(line.strip())
    return listWords


def stem(word):
    global stem_func, stem_enable, stem_lang
    if stem_enable and stem_func is not None:
        word = stem_func(word, stem_lang)
    return word


def get_all_words(text):
    text = text.lower()
    all_words = []
    for i in re.findall("[a-zа-я\d]+", text):
        all_words.append(stem(i))
    return all_words


def probability_text(count_ratio, pr_est, count_p, count_all):
    pr = 1
    for i in pr_est:
        pr *= (i + 1) / (count_p + count_all)
    pr *= count_ratio
    return pr


def get_counters(df, column, conditions=None):
    if conditions is None:
        conditions = [1, 0]

    global index_class1, index_class2
    return [
        df[column].count(),
        df[column].loc[df[column] == conditions[index_class1]].count(),
        df[column].loc[df[column] == conditions[index_class2]].count()
    ]


def test_classifier(df_check, words, counter=None, debug=None):
    global class_title, index_class1, index_class2

    if debug == True or debug == False:
        global show_debug
        show_debug = debug
    if counter is None:
        counter = []

    count_class1, count_class2, count_all = counter

    cnt_good, cnt_bad = [0, 0], [0, 0]
    # words_work = words_all
    for index, row in df_check.iterrows():
        text = row['text']
        debug_print(text)

        pr_est_class1 = []
        pr_est_class2 = []

        for word in get_all_words(text):
            if word in words:
                pr_est_class1.append(words[word][index_class1])
                pr_est_class2.append(words[word][index_class2])

        prob_class1 = probability_text(count_class1 / count_all, pr_est_class1, count_class1, len(words))
        prob_class2 = probability_text(count_class2 / count_all, pr_est_class2, count_class2, len(words))
        debug_print(f'Probability {class_title[index_class1]}: {prob_class1}')
        debug_print(f'Probability {class_title[index_class2]}: {prob_class2}')

        if prob_class1 > prob_class2:
            debug_print('Calc: {0} {1:.2f}%'.format(class_title[index_class1],
                                                    prob_class1 * 100 / (prob_class2 + prob_class1)))
            if row['class1'] == 1:
                debug_print(f'Sample: {class_title[index_class1]}, ', end='')
                debug_print('Good')
                cnt_good[index_class1] += 1
            else:
                debug_print(f'Sample: {class_title[index_class2]}, ', end='')
                debug_print('Bad')
                cnt_bad[index_class1] += 1

        else:
            debug_print(
                'Calc: {0} {1:.2f}%'.format(class_title[index_class2], prob_class2 * 100 / (prob_class2 + prob_class1)))
            if row['class1'] == 1:
                debug_print(f'Sample: {class_title[index_class1]}, ', end='')
                debug_print('Bad')
                cnt_bad[index_class2] += 1
            else:
                debug_print(f'Sample: {class_title[index_class2]}, ', end='')
                debug_print('Good')
                cnt_good[index_class2] += 1
        debug_print()
    return cnt_good, cnt_bad


def print_result(cnt_good, cnt_bad):
    global class_title, index_class1, index_class2
    print('Результат:')
    print('Тип {0}: попаданий ({1}), ошибок({2})'.format(
        class_title[index_class1],
        cnt_good[index_class1],
        cnt_bad[index_class1]))
    print('Тип {0}: попаданий ({1}), ошибок({2})'.format(
        class_title[index_class2],
        cnt_good[index_class2],
        cnt_bad[index_class2]))
    print('Всего, попаданий: {0}, ошибок: {1}'.format(cnt_good[index_class1] + cnt_good[index_class2],
                                                      cnt_bad[index_class1] + cnt_bad[index_class2]))
    print()


def calculate_words(df, stop_words):
    global index_class1, index_class2
    words_all = defaultdict(lambda: [0, 0])
    words_without_stop = defaultdict(lambda: [0, 0])

    for index, row in df.iterrows():
        for word in get_all_words(row['text']):
            # if stem_func is not None:
            #     word = stem_func(word)
            if row['class1'] == 1:
                i = index_class1
            else:
                i = index_class2
            words_all[word][i] += 1

            if word not in stop_words:
                words_without_stop[word][i] += 1

    return [words_all, words_without_stop]
