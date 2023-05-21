import numpy as np

def plot_data_distribution(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df['doc_len'] = df['CS_NAME'].apply(lambda words: len(words.split()))
    max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)
    sns.distplot(df['doc_len'], hist=True, kde=True, color='b', label='doc len')
    plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
    plt.title('plot length')
    plt.legend()
    plt.show()


def fetch_course_info(dataframe_idx, df):
    info = df.iloc[dataframe_idx]
    meta_dict = {'CS_NAME': info['CS_NAME'], 'CS_DESC_LONG': info['CS_DESC_LONG']}
    return meta_dict


def run_query_tests(model):
    print("Running query tests")
    query_test('Python Entwicklung', model)
    query_test('DevOps Azure CI/CD', model)


def query_test(query, model):
    results = model.search(query, top_k=5)

    print("\n")
    for result in results:
        print('\t', result)