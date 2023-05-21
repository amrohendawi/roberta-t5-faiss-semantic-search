import argparse
from zipfile import ZipFile
from tqdm import tqdm
import time
import os
import random
import gc
import pandas as pd

from sentence_transformers import SentenceTransformer

import faiss
import torch

from utils import *


class SemanticSearch:
    def __init__(self, model_name, local_model=False, device='cpu', data_file=None, columns=None, duplicate_filtering_columns=None, plot_distribution=False):
        print("Torch CUDA available: {}".format(torch.cuda.is_available()))
        self.data = pd.DataFrame()
        self.paragraphs = []
        self.model = None
        self.index = None
        self.device = device

        self.read_data(plot_distribution=plot_distribution, data_file=data_file, columns=columns, duplicate_filtering_columns=duplicate_filtering_columns)

        if local_model:
            self.load_local_model(model_name)
        else:
            self.fine_tune(model_name, generated_queries_file=f'models/generated_queries_{model_name}.tsv')

    def search(self, query, top_k):
        t = time.time()
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        print('>>>> Results in Total Time: {}'.format(time.time() - t))
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [fetch_course_info(idx, self.data) for idx in top_k_ids]
        return results

    def read_data(self, plot_distribution=False, data_file=None, columns=None, duplicate_filtering_columns=None):
        data = pd.read_json(data_file, encoding='utf-8')
        data.info()
        self.data = data[columns]
        del data
        gc.collect()

        self.data.dropna(inplace=True)
        self.data.drop_duplicates(subset=duplicate_filtering_columns, inplace=True)
        self.paragraphs = self.data.CS_DESC_LONG.tolist()

        if plot_distribution:
            plot_data_distribution(self.data)

    def load_local_model(self, model_folder):
        if not os.path.exists(f"models/{model_folder}") and os.path.exists(f'models/{model_folder}.zip'):
            with ZipFile(f'models/{model_folder}.zip', 'r') as zip_file:
                zip_file.extractall()
            print(f'models/{model_folder}.zip found and unzipped')
        self.model = SentenceTransformer(f"models/{model_folder}")
        self.model.to(self.device)

    def fine_tune(self, model_name, batch_size=10, num_queries=3, max_length_paragraph=512, max_length_query=48,
                  generated_queries_file=None):
        print("Fine tuning a new model")
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from sentence_transformers import InputExample, losses, models, datasets

        def generate_synthetic_queries(paragraphs, tsv):
            print("Generating synthetic queries")
            query_generator_model = "BeIR/query-gen-msmarco-t5-large-v1"
            stored_model = "models/query_generator_model"
            if not os.path.exists(stored_model):
                os.makedirs(stored_model)
                tokenizer = T5Tokenizer.from_pretrained(query_generator_model)
                generating_model = T5ForConditionalGeneration.from_pretrained(query_generator_model)
                generating_model.save_pretrained(stored_model)
                tokenizer.save_pretrained(stored_model)
            else:
                tokenizer = T5Tokenizer.from_pretrained(stored_model)
                generating_model = T5ForConditionalGeneration.from_pretrained(stored_model)

            generating_model.eval()
            generating_model.to(self.device)
            with open(tsv, 'w', encoding='utf-8') as fOut:
                for start_idx in tqdm(range(0, len(paragraphs), batch_size), desc="Generate queries", leave=False):
                    sub_paragraphs = paragraphs[start_idx:start_idx + batch_size]
                    inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph,
                                                             truncation=True, return_tensors='pt').to(self.device)
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

        if not os.path.exists(generated_queries_file):
            generate_synthetic_queries(paragraphs=self.paragraphs, tsv=generated_queries_file)

        if not os.path.exists(f'models/{model_name}') or not os.listdir(f'models/{model_name}'):
            run_fine_tune(tsv=f'models/generated_queries_{model_name}.tsv')

        # zip the new model in search folder
        with ZipFile(f'models/{model_name}.zip', 'w') as zipObj:
            zipObj.write(f'models/{model_name}')

    def create_index(self, index_file=None, data_column=None):
        print("Creating index")
        encoded_data = self.model.encode(self.data[data_column].tolist(), show_progress_bar=True)
        encoded_data = np.asarray(encoded_data.astype('float32'))
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.data))).astype(np.int64))
        faiss.write_index(self.index, index_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--model_name', type=str, default='rtx_3070', help='Name of the model')
    parser.add_argument('--local_model', action='store_true', help='Use a local pre-trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='weiterbildung', help='Path to the dataset folder where dataset.json is stored')
    parser.add_argument('--columns', nargs='+', default=['CS_NAME', 'CS_DESC_LONG'], help='Columns of the dataset to be used for the training')
    parser.add_argument('--dupl_filter_cols', nargs='+', default=['CS_DESC_LONG'], help='Columns of the dataset to be used for the duplicates filtering')
    parser.add_argument('--plot_distribution', action='store_true', help='Plot the distribution of the data')
    args = parser.parse_args()

    model = SemanticSearch(
        model_name=args.model_name,
        local_model=args.local_model,
        device=args.device,
        data_file=f"datasets/{args.dataset}/dataset.json",
        columns=args.columns,
        duplicate_filtering_columns=args.dupl_filter_cols,
        plot_distribution=args.plot_distribution
    )
    model.create_index(index_file=f"datasets/{args.dataset}/index.faiss", data_column=args.columns[0])
    run_query_tests(model)
