import pandas as pd
import numpy as np
import naive_bayes as nb

import pprint
pp = pprint.PrettyPrinter(indent=4)


def get_stem(word, lang=None):
    if lang is None:
        lang = 'en'
    return word


def normal_distribution():
    print('```')
    print('``` normal_distribution Example')
    print('```')
    print()
    data_golf_humidity = {'yes': [86, 96, 80, 65, 70, 80, 70, 90, 75], 'no': [85, 90, 70, 95, 91]}

    print('```Когда будем играть в гольф?```')
    print(f'Play yes, when humidity: {data_golf_humidity["yes"]} ')
    print(f'Play no, when humidity: {data_golf_humidity["no"]} ')
    print()
    humidity = [74, 86, 95]
    nb_nd = nb.NaiveBayesGaussian()
    threshold_value = 0.5
    print(f'Threshold value: {threshold_value}')
    for i in humidity:
        print(f'When humidity={i}?')
        pr = nb_nd.probability(i, main_data=data_golf_humidity['yes'], opponent_data=data_golf_humidity['no'])
        print(f'p: {pr} - ', end='')
        if pr > threshold_value:
            print(f'Yes!')
        else:
            print(f'No!')
        print()


# def ham_spam():
#     # # Байесовский классификатор для текстов
#
#     # ## Иницилизация языковой библиотеки для работы с корнем
#
#     # In[ ]:
#
#     def get_stem(word, lang=None):
#         if lang is None:
#             lang = 'en'
#         return word
#
#     nb_text.stem_func = get_stem
#     nb_text.stem_enable = True
#     nb_text.stem_lang = 'en'
#
#     # ## Коты/Собаки
#     # Проверим алгоритм на задаче из лекции, про котов и собак
#
#     # ### Загрузка и подготовка данных
#
#     # In[ ]:
#
#     nb_text.stem_enable = True
#     nb_text.stem_lang = 'ru'
#
#     nb_text.class_title[nb_text.index_class1] = 'Коты'
#     nb_text.class_title[nb_text.index_class2] = 'Собаки'
#
#     lst_text = [
#         'Кот бежит к будке и говорит мяу.',
#         'Белого кота и чёрного кота несут в котоноске.',
#         'Большой кот и маленький кот поймали мышь.',
#         'Собака из будки смотрит на кота.',
#         'Собака залезла не в свою будку, а в чужую будку.',
#     ]
#     lst_class = [1, 1, 1, 0, 0]
#     df = pd.DataFrame(list(zip(lst_class, lst_text)),
#                       columns=['class1', 'text'])
#     df_test = pd.DataFrame(list(zip([1], ['Белый кот, чёрный кот и рыжий кот идут мимо будки собаки'])),
#                            columns=['class1', 'text'])
#
#     # In[ ]:
#
#     count_all, count_cats, count_dogs = nb_text.get_counters(df=df, column='class1')
#
#     print(
#         f'Данные для расчета: {nb_text.class_title[nb_text.index_class1]}: {count_cats}, {nb_text.class_title[nb_text.index_class2]}: {count_dogs}')
#     print('Данные для проверки: {0}: {1}, {2}: {3}'.format(
#         nb_text.class_title[nb_text.index_class1],
#         df_test['class1'].loc[df_test['class1'] == 1].count(),
#         nb_text.class_title[nb_text.index_class2],
#         df_test['class1'].loc[df_test['class1'] == 0].count()
#     ))
#
#     # In[ ]:
#
#     stop_words = nb_text.load_stop_words('input/stopwords_ru.txt')
#
#     # ### Подсчет слов
#
#     # In[ ]:
#
#     words_all, words_without_stop = nb_text.calculate_words(df=df, stop_words=stop_words)
#
#     print('All - {0}, without stop words - {1}'.format(len(words_all), len(words_without_stop)))
#
#     # ### Проверка на тестовой выборке
#
#     # #### с учетом стоп слов
#
#     # In[ ]:
#
#     ### test data procces
#     print('С учетом стоп слов')
#     cnt_good, cnt_bad = nb_text.test_classifier(
#         df_check=df_test,
#         words=words_without_stop,
#         counter=[count_cats, count_dogs, count_all],
#         debug=False
#     )
#     nb_text.print_result(cnt_good, cnt_bad)
#
#     # #### без учета стоп слов
#
#     # In[ ]:
#
#     ### test data procces
#     print('Без учета стоп слов')
#     cnt_good, cnt_bad = nb_text.test_classifier(
#         df_check=df_test,
#         words=words_all,
#         counter=[count_cats, count_dogs, count_all],
#         debug=False
#     )
#     nb_text.print_result(cnt_good, cnt_bad)
#
#     # ## Spam/Ham
#
#     # ### Загрузка и подготовка данных
#
#     # In[ ]:
#
#     nb_text.stem_enable = True
#     nb_text.stem_lang = 'en'
#
#     nb_text.class_title[nb_text.index_class1] = 'Spam'
#     nb_text.class_title[nb_text.index_class2] = 'Ham'
#
#     fraction_test = 0.1
#
#     df_raw = pd.read_csv('input/SMSSpamCollection', delimiter='\t', header=None, names=['class1', 'text'])
#     df_raw['class1'] = np.where(df_raw['class1'] == "spam", 1, 0)
#
#     df_test = pd.concat([
#         df_raw.loc[df_raw['class1'] == 1].sample(frac=fraction_test, random_state=12345),
#         df_raw.loc[df_raw['class1'] == 0].sample(frac=fraction_test, random_state=12345)
#     ])
#     df = df_raw.drop(df_test.index)
#
#     # In[ ]:
#
#     count_all, count_spam, count_ham = nb_text.get_counters(df=df, column='class1')
#
#     print(
#         f'Данные для расчета: {nb_text.class_title[nb_text.index_class1]}: {count_spam}, {nb_text.class_title[nb_text.index_class2]}: {count_ham}')
#     print('Данные для проверки: {0}: {1}, {2}: {3}'.format(
#         nb_text.class_title[nb_text.index_class1],
#         df_test['class1'].loc[df_test['class1'] == 1].count(),
#         nb_text.class_title[nb_text.index_class2],
#         df_test['class1'].loc[df_test['class1'] == 0].count()
#     ))
#
#     # In[ ]:
#
#     stop_words = nb_text.load_stop_words('input/stopwords_en.txt')
#
#     # ### Подсчет слов
#
#     # In[ ]:
#
#     words_all, words_without_stop = nb_text.calculate_words(df=df, stop_words=stop_words)
#
#     print('All - {0}, without stop words - {1}'.format(len(words_all), len(words_without_stop)))
#
#     # ### Проверка на тестовой выборке
#
#     # #### с учетом стоп слов
#
#     # In[ ]:
#
#     ### test data procces
#     print('С учетом стоп слов')
#     cnt_good, cnt_bad = nb_text.test_classifier(
#         df_check=df_test,
#         words=words_without_stop,
#         counter=[count_spam, count_ham, count_all],
#         debug=False
#     )
#     nb_text.print_result(cnt_good, cnt_bad)
#
#     # #### без учета стоп слов
#
#     # In[ ]:
#
#     ### test data procces
#     print('Без учета стоп слов')
#     cnt_good, cnt_bad = nb_text.test_classifier(
#         df_check=df_test,
#         words=words_all,
#         counter=[count_spam, count_ham, count_all],
#         debug=False
#     )
#     nb_text.print_result(cnt_good, cnt_bad)


def run_cats_dogs():
    cats_dogs = nb.NaiveBayesMultinomial()

    # # Байесовский классификатор для текстов

    # ## Иницилизация языковой библиотеки для работы с корнем

    # In[ ]:

    cats_dogs.stem_func = get_stem
    cats_dogs.stem_enable = True
    cats_dogs.stem_lang = 'ru'
    # ## Коты/Собаки
    # Проверим алгоритм на задаче из лекции, про котов и собак

    # ### Загрузка и подготовка данных

    # In[ ]:

    cats_dogs.set_title('Коты', 'Собаки')

    lst_text = [
        'Кот бежит к будке и говорит мяу.',
        'Белого кота и чёрного кота несут в котоноске.',
        'Большой кот и маленький кот поймали мышь.',
        'Собака из будки смотрит на кота.',
        'Собака залезла не в свою будку, а в чужую будку.',
    ]
    lst_class = [1, 1, 1, 0, 0]
    df = pd.DataFrame(list(zip(lst_class, lst_text)),
                      columns=['class1', 'text'])
    df_test = pd.DataFrame(list(zip([1], ['Белый кот, чёрный кот и рыжий кот идут мимо будки собаки'])),
                           columns=['class1', 'text'])

    # In[ ]:

    count_all, count_cats, count_dogs = cats_dogs.calculate_counters(df=df, column='class1')

    print(
        f'Данные для расчета: {cats_dogs.class_title[cats_dogs.index_class1]}: {count_cats}, {cats_dogs.class_title[cats_dogs.index_class2]}: {count_dogs}')
    print('Данные для проверки: {0}: {1}, {2}: {3}'.format(
        cats_dogs.class_title[cats_dogs.index_class1],
        df_test['class1'].loc[df_test['class1'] == 1].count(),
        cats_dogs.class_title[cats_dogs.index_class2],
        df_test['class1'].loc[df_test['class1'] == 0].count()
    ))
    # In[ ]:

    cats_dogs.load_stop_words('input/stopwords_ru.txt')

    # ### Подсчет слов

    # In[ ]:

    cats_dogs.calculate_words(df=df)

    print('All - {0}, without stop words - {1}'.format(len(cats_dogs.words_all), len(cats_dogs.words_without_stop)))

    # ### Проверка на тестовой выборке

    # #### с учетом стоп слов

    # In[ ]:

    ### test data procces
    print('Текст "{0}"'.format(
        df_test.iloc[0]['text']
    ))

    print('### С учетом стоп слов ###')
    print('входит в группу "{0}" с вероятностью {1}'.format(
        cats_dogs.class_title[cats_dogs.index_class1],
        cats_dogs.probability(text=df_test.iloc[0]['text'], mode=cats_dogs.index_class1,
                              words=cats_dogs.words_without_stop)
    ))
    print('входит в группу "{0}" с вероятностью {1}'.format(
        cats_dogs.class_title[cats_dogs.index_class2],
        cats_dogs.probability(text=df_test.iloc[0]['text'], mode=cats_dogs.index_class2,
                              words=cats_dogs.words_without_stop)
    ))

    # #### без учета стоп слов

    # In[ ]:

    ### test data procces
    print('Без учета стоп слов')
    print('входит в группу "{0}" с вероятностью {1}'.format(
        cats_dogs.class_title[cats_dogs.index_class1],
        cats_dogs.probability(text=df_test.iloc[0]['text'], mode=cats_dogs.index_class1,
                              words=cats_dogs.words_all)
    ))
    print('входит в группу "{0}" с вероятностью {1}'.format(
        cats_dogs.class_title[cats_dogs.index_class2],
        cats_dogs.probability(text=df_test.iloc[0]['text'], mode=cats_dogs.index_class2,
                              words=cats_dogs.words_all)
    ))


def run_spam_ham():
    # ## Spam/Ham

    # ### Загрузка и подготовка данных

    # In[ ]:

    spam_ham = nb.NaiveBayesMultinomial()
    spam_ham.stem_func = get_stem
    spam_ham.stem_enable = True
    spam_ham.stem_lang = 'en'

    spam_ham.set_title('Spam', 'Ham')

    fraction_test = 0.1

    df_raw = pd.read_csv('input/SMSSpamCollection', delimiter='\t', header=None, names=['class1', 'text'])
    df_raw['class1'] = np.where(df_raw['class1'] == "spam", 1, 0)

    df_test = pd.concat([
        df_raw.loc[df_raw['class1'] == 1].sample(frac=fraction_test, random_state=12345),
        df_raw.loc[df_raw['class1'] == 0].sample(frac=fraction_test, random_state=12345)
    ])
    df = df_raw.drop(df_test.index)
    # In[ ]:

    count_all, count_spam, count_ham = spam_ham.calculate_counters(df=df, column='class1')

    print(
        f'Данные для расчета: {spam_ham.class_title[spam_ham.index_class1]}: {count_spam}, {spam_ham.class_title[spam_ham.index_class2]}: {count_ham}')
    print('Данные для проверки: {0}: {1}, {2}: {3}'.format(
        spam_ham.class_title[spam_ham.index_class1],
        df_test['class1'].loc[df_test['class1'] == 1].count(),
        spam_ham.class_title[spam_ham.index_class2],
        df_test['class1'].loc[df_test['class1'] == 0].count()
    ))
    # In[ ]:

    spam_ham.load_stop_words('input/stopwords_en.txt')
    # ### Подсчет слов

    # In[ ]:

    words_all, words_without_stop = spam_ham.calculate_words(df=df)

    print('All - {0}, without stop words - {1}'.format(len(words_all), len(words_without_stop)))
    # ### Проверка на тестовой выборке

    # #### с учетом стоп слов

    # In[ ]:

    ### test data procces
    print('С учетом стоп слов')
    cnt_good, cnt_bad = spam_ham.test_classifier(
        df_check=df_test,
        words=words_without_stop,
        debug=False
    )
    spam_ham.print_result(cnt_good, cnt_bad)

    # #### без учета стоп слов

    # In[ ]:

    ### test data procces
    print('Без учета стоп слов')
    cnt_good, cnt_bad = spam_ham.test_classifier(
        df_check=df_test,
        words=words_all,
        debug=False
    )
    spam_ham.print_result(cnt_good, cnt_bad)


def run_multinomial():
    # 'sports' == 1
    # 'suv' == 2
    train_data = [
        ['red', 'sports', 'domestic', 1],
        ['red', 'sports', 'domestic', 0],
        ['red', 'sports', 'domestic', 1],
        ['yellow', 'sports', 'domestic', 0],
        ['yellow', 'sports', 'imported', 1],
        ['yellow', 'suv', 'imported', 0],
        ['yellow', 'suv', 'imported', 1],
        ['yellow', 'suv', 'domestic', 0],
        ['red', 'suv', 'imported', 0],
        ['red', 'sports', 'imported', 1],
    ]

    ## prepare
    print(isinstance(1, int))
    df_train = pd.DataFrame(list(train_data), columns=['color', 'type', 'origin', 'stolen'])
    df_train['stolen'] = np.where(df_train['stolen'] == 1, 'yes', 'no')
    ####

    print(df_train.head(5))
    print()

    multinomial = nb.NaiveBayesCategorical()
    multinomial.train(df=df_train, class_column='stolen')
    # pp.pprint(multinomial.get_likelihood_table())
    # return

    data_check = {'color': 'red', 'type': 'suv', 'origin': 'domestic'}
    p = multinomial.probability(data_check=data_check)

    print()
    print(f'{p["answer"]} {data_check}?')

    answer = {'title': '', 'p': 0}
    for i in p['probability']:
        if answer['title'] == '' or p['probability'][i] >= answer['p']:
            answer['title'] = i
            answer['p'] = p['probability'][i]
    print('{0}: {1}'.format(answer['title'], answer['p']))

    print()
    print('raw probability: {0}'.format(p['probability']))

    return


if __name__ == '__main__':
    # normal_distribution()
    # run_cats_dogs()
    # spam_ham()
    run_multinomial()
