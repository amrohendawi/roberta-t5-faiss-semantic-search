import argparse
from SemanticSearch import SemanticSearch
from utils import run_query_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Search')
    parser.add_argument('--model_name', type=str, default='rtx_3070', help='Name of the model')
    parser.add_argument('--local_model', action='store_true', help='Use a local pre-trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='weiterbildung', help='Path to the dataset folder where dataset.json is stored')
    parser.add_argument('--title_col', type=str, default='CS_NAME', help='Title column of the dataset to be used for the training and plotting distribution')
    parser.add_argument('--desc_col', type=str, default='CS_DESC_LONG', help='Column of the dataset to be used for the training and filtering duplicates')
    parser.add_argument('--plot_distribution', action='store_true', help='Plot the distribution of the data')
    args = parser.parse_args()

    model = SemanticSearch(
        model_name=args.model_name,
        local_model=args.local_model,
        device=args.device,
        data_file=f"datasets/{args.dataset}/dataset.json",
        title_col=args.title_col,
        desc_col=args.desc_col,
        plot_distribution=args.plot_distribution
    )
    model.create_index(index_file=f"datasets/{args.dataset}/index.faiss", data_column=args.columns[0])
    run_query_tests(model, desc_col=args.desc_col, title_col=args.title_col)
