import pandas as pd
from dash import Dash, html, Input, Output, State, callback, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from SemanticSearch import SemanticSearch

app = Dash(
    __name__, meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "DE Semantic Search"

app.config.suppress_callback_exceptions = True

server = app.server

data = pd.read_json('courses_dataset.json', encoding='utf-8')
data = data[['CS_NAME', 'CS_DESC_LONG']]
data.dropna(inplace=True)
data.drop_duplicates(subset=['CS_NAME'], inplace=True)
data.info()

model = None
if not model:
    model = SemanticSearch(model_name='rtx_3070', local_model=True)
    model.create_index()

app.layout = dmc.Center(
    dmc.Group(
        children=[
            dmc.Header(
                children=[
                    html.H4("RoBERTA+T5+FAISS Semantic Search", className="display-6"),
                    html.P("""
                        This is a demonstration of a semantic search implementation based on RoBERTA model fine-tuned
                        on a 16000+ training courses dataset.
                        """,
                           ),
                ]
            ),
            dmc.Group(
                children=[
                    dmc.TextInput(
                        type="search",
                        id="search-input",
                        placeholder="Search for courses",
                    ),
                    dmc.Slider(
                        id="drag-slider",
                        updatemode="drag",
                        min=1,
                        max=20,
                        step=1,
                        value=10,
                        marks=
                        [
                            {"value": 1, "label": "1"},
                            {"value": 10, "label": "10"},
                            {"value": 20, "label": "20"},
                        ],
                        color="teal",
                        size="lg",
                        style={"minWidth": "200px"},
                    ),
                ],
                spacing="md",
                align="stretch",
                position="apart",
            ),
            dmc.Accordion(
                id="accordion",
                multiple=True,
            )

        ],
        direction="column",
        spacing="md",
        style={"maxWidth": "70%", "marginTop": "2rem"},
        align="center",
    )
)


@callback(
    Output("accordion", "children"),
    [Input("search-input", "value"),
     State("drag-slider", "value"),
     ],
)
def update_accordion(query, slider_value):
    if model and query:
        results = model.search(query, top_k=slider_value)
        results = pd.DataFrame(results)
        return entries_as_accordion_items(results)
    return entries_as_accordion_items(data[:10])


def entries_as_accordion_items(entries):
    return [dmc.AccordionItem(
        dcc.Markdown(
            entries.iloc[i]['CS_DESC_LONG']
        ),
        label=entries.iloc[i]['CS_NAME'],
    ) for i in range(len(entries))]


if __name__ == "__main__":
    app.run_server(debug=True)
