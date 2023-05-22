import numpy as np


def plot_data_distribution(df, title_col=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df['doc_len'] = df[title_col].apply(lambda words: len(words.split()))
    max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)
    sns.distplot(df['doc_len'], hist=True, kde=True, color='b', label='doc len')
    plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
    plt.title('plot length')
    plt.legend()
    plt.show()


def fetch_course_info(dataframe_idx, df, title_col=None, desc_col=None):
    info = df.iloc[dataframe_idx]
    meta_dict = {title_col: info[title_col], desc_col: info[desc_col]}
    return meta_dict


def run_query_tests(model, desc_col=None, title_col=None):
    print("Running query tests")
    query_test('Python Entwicklung', model, desc_col=desc_col, title_col=title_col)
    query_test('DevOps Azure CI/CD', model, desc_col=desc_col, title_col=title_col)


def query_test(query, model, desc_col=None, title_col=None):
    results = model.search(query, top_k=5, desc_col=desc_col, title_col=title_col)

    print("\n")
    for result in results:
        print('\t', result)
