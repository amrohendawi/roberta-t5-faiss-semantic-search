from zipfile import ZipFile
from tqdm import tqdm
import time
import os
import random
import gc
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

import faiss
import torch


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


def query_test(query, model):
    results = model.search(query, top_k=5)

    print("\n")
    for result in results:
        print('\t', result)


class SemanticSearch:
    def __init__(self, model_name, local_model=False):
        print("Torch CUDA available: {}".format(torch.cuda.is_available()))
        self.data = pd.DataFrame()
        self.paragraphs = []
        self.model = None
        self.index = None
        self.read_data()

        if local_model:
            self.load_local_model(model_name)
        else:
            self.fine_tune(model_name)

    def search(self, query, top_k):
        t = time.time()
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        print('>>>> Results in Total Time: {}'.format(time.time() - t))
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [fetch_course_info(idx, self.data) for idx in top_k_ids]
        return results

    def read_data(self, plot_distribution=False):
        data = pd.read_json('courses_dataset.json', encoding='utf-8')
        data.info()
        self.data = data[['CS_NAME', 'CS_DESC_LONG']]
        del data
        gc.collect()

        self.data.dropna(inplace=True)
        self.data.drop_duplicates(subset=['CS_DESC_LONG'], inplace=True)
        self.paragraphs = self.data.CS_DESC_LONG.tolist()

        if plot_distribution:
            plot_data_distribution(self.data)

    def load_local_model(self, model_name):
        if not os.path.exists(f'models/{model_name}') and os.path.exists(f'{model_name}.zip'):
            with ZipFile(f'{model_name}.zip', 'r') as zip_file:
                zip_file.extractall()
            print(f'{model_name}.zip found and unzipped')
        self.model = SentenceTransformer(f'models/{model_name}')
        self.model.to('cuda')

    def fine_tune(self, model_name, batch_size=2, num_queries=3, max_length_paragraph=512, max_length_query=48):
        print("Fine tuning a new model")
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from sentence_transformers import InputExample, losses, models, datasets

        def generate_synthetic_queries(paragraphs, tsv, device='cuda'):
            tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
            generating_model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
            generating_model.eval()
            generating_model.to(device)
            with open(tsv, 'w', encoding='utf-8') as fOut:
                for start_idx in tqdm(range(0, len(paragraphs), batch_size)):
                    sub_paragraphs = paragraphs[start_idx:start_idx + batch_size]
                    inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph,
                                                             truncation=True, return_tensors='pt').to(device)
                    outputs = generating_model.generate(
                        **inputs,
                        max_length=max_length_query,
                        do_sample=True,
                        top_p=0.95,
                        num_return_sequences=num_queries)

                    for idx, out in enumerate(outputs):
                        query = tokenizer.decode(out, skip_special_tokens=True, encoding='utf-8')
                        para = sub_paragraphs[int(idx / num_queries)]
                        fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

        def run_fine_tune(tsv, num_epochs=3):
            train_examples = []
            with open(tsv, encoding='utf-8') as fIn:
                for line in fIn:
                    try:
                        query, paragraph = line.strip().split('\t', maxsplit=1)
                        train_examples.append(InputExample(texts=[query, paragraph]))
                    except:
                        pass

            random.shuffle(train_examples)
            train_examples = train_examples[:2000]
            # For the MultipleNegativesRankingLoss, it is important that the batch does not contain duplicate entries,
            # i.e. no two equal queries and no two equal paragraphs. To ensure this, we use a special data loader
            train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=4)

            # Now we create a SentenceTransformer model from scratch
            word_emb = models.Transformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
            pooling = models.Pooling(word_emb.get_word_embedding_dimension())
            self.model = SentenceTransformer(modules=[word_emb, pooling])

            # MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
            # and trains the model so that it is suitable for semantic search
            train_loss = losses.MultipleNegativesRankingLoss(self.model)

            warmup_steps = int(len(train_dataloader) * num_epochs * 0.05)
            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs,
                           warmup_steps=warmup_steps,
                           use_amp=False, show_progress_bar=True)

            os.makedirs(f'models/{model_name}', exist_ok=True)
            self.model.save(f'models/{model_name}')

        if not os.path.exists(f'generated_queries_{model_name}.tsv'):
            generate_synthetic_queries(self.paragraphs, tsv=f'generated_queries_{model_name}.tsv')

        if not os.path.exists(f'models/{model_name}') or not os.listdir(f'models/{model_name}'):
            run_fine_tune(tsv=f'generated_queries_{model_name}.tsv')

        # zip the new model in search folder
        with ZipFile(f'{model_name}.zip', 'w') as zipObj:
            zipObj.write(f'models/{model_name}')

    def create_index(self):
        print("Creating index")
        encoded_data = self.model.encode(self.data.CS_NAME.tolist(), show_progress_bar=True)
        encoded_data = np.asarray(encoded_data.astype('float32'))
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.data))).astype(np.int64))
        faiss.write_index(self.index, 'course_description.index')

    def run_query_tests(self):
        print("Running query tests")
        query_test('Python Entwicklung', self.model)
        query_test('DevOps Azure CI/CD', self.model)

if __name__ == '__main__':
    model = SemanticSearch(model_name='rtx_3070', local_model=True)
    model.create_index()
    model.run_query_tests()
