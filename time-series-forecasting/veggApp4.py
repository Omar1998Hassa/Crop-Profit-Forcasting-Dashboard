import os
import time
import cmdstanpy
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    from https://github.com/facebook/prophet/issues/223

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def run_prophet(
    df,
    weekly_seasonality=False,
    start=2012,
    end=2016,
    uncertainty_samples=200,
    periods=180,
):
    yr = pd.to_datetime(df.ds).dt.year
    yr_df = df[(yr >= start) & (yr <= end)]
    with suppress_stdout_stderr():
        m = Prophet(
            weekly_seasonality=weekly_seasonality,
            uncertainty_samples=uncertainty_samples,
            growth="flat",
        )
        m.fit(yr_df)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    return m, forecast

def NamedCardGroup(children, label, **kwargs):
    return dbc.Card(
        [
            dbc.CardHeader(label),
            dbc.CardBody(children)
        ],
        **kwargs
    )

value2label = {
    "FOODS": "Business",
    "HOBBIES": "Other",
    "HOUSEHOLD": "Residential",
    "CA": "California",
    "TX": "Texas",
    "WI": "Wisconsin",
}

veggDf = pd.read_csv(r"C:\Users\Omar\Downloads\VisualData2project\scale-ai-templates-master\scale-ai-templates-master\apps\time-series-forecasting\archive\ProductPriceIndex.csv")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])
server = app.server  # expose server variable for Procfile


data_controls = [
    NamedCardGroup(
    dcc.Dropdown(
        id="radio-category",
        options=[
            {"label": productname, "value": productname}
            for productname in veggDf["productname"].unique()
        ],
        value=veggDf["productname"].iloc[0],  # Set default value to the first unique product name
        clearable=False,  # Optional: prevents the dropdown from being cleared
        ),
        label="Product",
    ),
    NamedCardGroup(
        dcc.Dropdown(
            id="radio-state",
            options=[
                {"label": "farm Price", "value": "farmprice"},
                {"label": "Chicago Retail", "value": "chicagoretail"},
                {"label": "Los Angeles Retail", "value": "losangelesretail"},
                {"label": "New York Retail", "value": "newyorkretail"},
            ],
            value="farmprice",  # Default value
            clearable=False,  # Optional: prevents the dropdown from being cleared
        ),
        label="Price",
    ),
    NamedCardGroup(
        dcc.RangeSlider(
            id="range-years",
            min=1999,
            max=2019,
            step=1,
            value=[2012, 2016],
            marks={
                year: {"label": str(year), "style": {"writing-mode": "vertical-rl"}}
                for year in range(1999,2020, 1)
            },
        ),
        label="Years Fitted",
    ),
]


model_controls = [
    dbc.CardHeader("Model Controls"),
    dbc.CardBody(
        [
            NamedCardGroup(
                dbc.RadioItems(
                    inline=True,
                    id="radio-seasonality",
                    options=[
                        {"label": "Enabled", "value": 1},
                        {"label": "Disabled", "value": 0},
                    ],
                    value=0,
                ),
                label="Weekly Seasonality",
            ),
            NamedCardGroup(
                dbc.FormGroup(
                    [
                        dbc.Label("Num Days Forecasted"),
                        dcc.Input(
                            id="slider-forecast-days",
                            type="number",
                            min=0,
                            max=2000,
                            step=5,
                            value=180,
                            style={"width": "100%"},
                        ),
                    ]
                ),
                label="Num Days Forecasted",
            ),
            NamedCardGroup(
                dcc.Slider(
                    id="slider-uncertainty-samples",
                    min=0,
                    max=600,
                    step=100,
                    value=200,
                    marks={i: str(i) for i in [0, 200, 400, 600]},
                ),
                label="Uncertainty Samples",
            ),
        ]
    )
]



app.layout = html.Div(
    children=[
        dbc.Container(
            [
                html.Div(
                    [
                        html.H1("Harvest Insights: Veggie & Fruit Trends Dashboard", style={'color': 'black', 'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
                    ],
                    style={'display': 'inline-block', 'padding': '10px', 'width': '100%', 'background-color': 'rgba(225, 225, 255, 0.6)', 'margin': '0 auto'}
                ),
                html.Hr(),
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Prediction Model",
                            children=[
                                dbc.Card(dbc.Row([dbc.Col(c) for c in data_controls]), body=True, style={'background-color': 'transparent', 'border': 'none'}),
                                 dbc.Row(
                                    [
                                        dbc.Col(dbc.Card(model_controls), md=3,
                                                style={'background-color': 'transparent', 'border': 'none'}),
                                        dbc.Col(dbc.Card(dcc.Graph(id="graph-forecast"),className="bg-transparent", body=False), md=5,
                                                style={'background-color': 'transparent', 'border': 'none'}),
                                        dbc.Col(dbc.Card(dcc.Graph(id="graph-components"),className="bg-transparent", body=False), md=4,
                                                style={'background-color': 'transparent', 'border': 'none'}),
                                    ]
                                ),
                            ],
                        ),
                         dbc.Tab(
            label="Fluctuations of crops prices during recorded years ",
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader("Fluctuations of crops prices during recorded years "),
                        dbc.CardBody(
                            [
                                html.Label("Select Price:", style={"display": "block"}),  # Add label
                                dcc.Dropdown(
                                    id='y-axis-checkbox',
                                    options=[
                                        {'label': 'Farm Price', 'value': 'farmprice'},
                                        {'label': 'Chicago Retail Price', 'value': 'chicagoretail'},
                                        {'label': 'Los Angeles Retail Price', 'value': 'losangelesretail'},
                                        {'label': 'New York Retail Price', 'value': 'newyorkretail'},
                                    ],
                                    value='farmprice',  # Default selection
                                    style={"margin-right": "10px"}  # Adjust style
                                ),
                                dcc.Graph(id='Box-plot-graph'),
                            ]
                        ),
                    ],
                    style={"width": "100%"}  # Adjust the width of the card
                ),
            ],
        ),
        dbc.Tab(
            label="Average Prices of Vegetables vs Fruits",
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader("Average Prices of Vegetables vs Fruits"),
                        dbc.CardBody(
                            [
                                dcc.Dropdown(
                                    id='price-type-radio',
                                    options=[
                                        {'label': 'Farm Price', 'value': 'farmprice'},
                                        {'label': 'Chicago Retail Price', 'value': 'chicagoretail'},
                                        {'label': 'Los Angeles Retail Price', 'value': 'losangelesretail'},
                                        {'label': 'New York Retail Price', 'value': 'newyorkretail'},
                                    ],
                                    value='farmprice',
                                    # inline=True
                                ),
                                dcc.Graph(id='average-price-plot')
                            ]
                        ),
                    ],
                    style={"width": "100%"}  # Set width to 100% of the page
                ),
            ],
        ),
        dbc.Tab(
            label="Correlation between Crop Prices",
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader("Correlation between Crop Prices"),
                        dbc.CardBody(
                            [
                                dcc.Dropdown(
                                    id='corr-radio',
                                    options=[
                                        {'label': 'Farm Price', 'value': 'farmprice'},
                                        {'label': 'Chicago Retail Price', 'value': 'chicagoretail'},
                                        {'label': 'Los Angeles Retail Price', 'value': 'losangelesretail'},
                                        {'label': 'New York Retail Price', 'value': 'newyorkretail'},
                                    ],
                                    value='farmprice',
                                    # inline=True
                                ),
                                dcc.Graph(id='corr-price-plot')
                            ]
                        ),
                    ],
                    style={"width": "100%"}  # Set width to 100% of the page
                ),
            ],
        ),
        dbc.Tab(
            label="Line Plots of Product Prices",
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader("Line Plots of Product Prices"),
                        dbc.CardBody(
                            [
                                dcc.Dropdown(
                                    id='price-category-dropdown',
                                    options=[
                                        {'label': 'Farm Price', 'value': 'farmprice'},
                                        {'label': 'Chicago Retail Price', 'value': 'chicagoretail'},
                                        {'label': 'Los Angeles Retail Price', 'value': 'losangelesretail'},
                                        {'label': 'New York Retail Price', 'value': 'newyorkretail'}
                                    ],
                                    value='farmprice',  # Default value
                                    style={'margin-bottom': '10px'}  # Add some margin below the dropdown
                                ),
                                dcc.Dropdown(
                                    id='product-dropdown',
                                    options=[{'label': product, 'value': product} for product in veggDf['productname'].unique()],
                                    value=[veggDf['productname'].unique()[0]],  # Default value
                                    multi=True,  # Enable multi-select
                                    style={'margin-bottom': '10px'}  # Add some margin below the dropdown
                                ),
                                dcc.Graph(id='product-price-plot')
                            ]
                        ),
                    ],
                    style={"width": "100%"}  # Set width to 100% of the page
                ),
            ],
        )
                    ]
                ),
            ],
            fluid=True,
            style={
                'background-image': 'url("/assets/fruits-and-vegetables-pattern-on-black-vector.avif")',
                'background-size': 'cover',
                'background-position': 'center',
                'height': '100vh',
                'padding': '20px'
            }
        )
    ]
)
# Define callback to update the plot
@app.callback(
    Output('product-price-plot', 'figure'),
    [Input('price-category-dropdown', 'value'),
     Input('product-dropdown', 'value')]
)
def update_plot(price_category, selected_products):
    # Filter dataframe based on selected products
    filtered_df = veggDf[veggDf['productname'].isin(selected_products)]
    
    # Create an empty figure
    fig = go.Figure()
    
    # Loop through each selected product and add it as a line plot
    for product in selected_products:
        product_data = filtered_df[filtered_df['productname'] == product]
        fig.add_trace(go.Scatter(x=product_data['date'], y=product_data[price_category], mode='lines', name=product))
    
    # Customize the layout of the plot
    fig.update_layout(
        title="Product Prices Over Time",
        xaxis_title="Date",
        yaxis_title=price_category.capitalize(),  # Dynamically set y-axis title based on selected price category
        hovermode='closest'
    )
    
    return fig
# def update_correlation_plot():
#     veggDf = pd.read_csv(r"C:\Users\Omar\Downloads\VisualData2project\scale-ai-templates-master\scale-ai-templates-master\apps\time-series-forecasting\archive\ProductPriceIndex.csv")

#     # Filter data based on selected product
#     veggDf['year'] = pd.to_datetime(veggDf['date']).dt.year
#     veggDf = veggDf.dropna(subset=['farmprice', 'chicagoretail', 'losangelesretail', 'newyorkretail'])

#     veggDf = veggDf[veggDf[:] !='$']
#     veggDf[:] = veggDf[:].replace('[\$,]', '', regex=True).astype(float)
    
#     # Group by year and calculate the correlation matrix
#     correlation_matrix = veggDf[['farmprice', 'chicagoretail', 'losangelesretail', 'newyorkretail']].corr()

#     # Create the correlation heatmap plot
#     fig = px.imshow(correlation_matrix,
#                     labels=dict(color="Correlation"),
#                     color_continuous_scale="coolwarm",
#                     width=None,
#                     height=None)
#     fig.update_layout(title='Correlation Between Different Prices')
    
#     return fig

# @app.callback(
#     dash.dependencies.Output('corr-price-plot-2', 'figure'),
#     []
# )
# def update_correlation():
#     fig = update_correlation_plot()
#     return fig

# @app.callback(
#     Output('corr-price-plot-2', 'figure')
# )
# def update_correlation_plot(selected_price_type):
#     # Select the columns corresponding to the selected price type
#     veggDf = pd.read_csv(r"C:\Users\Omar\Downloads\VisualData2project\scale-ai-templates-master\scale-ai-templates-master\apps\time-series-forecasting\archive\ProductPriceIndex.csv")

#     # # Filter data based on selected product
#     # veggDf['year'] = pd.to_datetime(veggDf['date']).dt.year
#     # # Convert 'price' column to numeric type, coerce non-numeric values to NaN
#     # veggDf = veggDf[veggDf[selected_price_type]!='$']
#     # veggDf[selected_price_type] = veggDf[selected_price_type].replace('[\$,]', '', regex=True).astype(float)

#     # # Drop rows with NaN values in 'price' column

#     # veggDf.dropna(subset=[selected_price_type], inplace=True)
    
#     # avg_prices = veggDf.groupby(['year', 'productname'])[selected_price_type].mean().reset_index()
    
#     # # Subset the dataframe with selected columns
#     # # price_data = avg_prices[selected_price_type]
#     # pivot_df = avg_prices.pivot(index='producname', columns='year', values=selected_price_type)

#     # Calculate the correlation matrix
#     correlation_matrix = veggDf[["farmprice","chicagoretail","newyorkretail","losangelesretail"]].corr()

#     # Create the correlation heatmap plot
#     fig = px.imshow(correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns,
#                     labels=dict(color="Correlation"), color_continuous_scale="Viridis")

#     fig.update_layout(title=f'Correlation Between {selected_price_type} of Different Crops',width=1800,  # Adjust the width of the plot
#                       height=600)

#     return fig

@app.callback(
    Output('corr-price-plot', 'figure'),
    [Input('corr-radio', 'value')]
)
def update_correlation_plot(selected_price_type):
    # Select the columns corresponding to the selected price type
    veggDf = pd.read_csv(r"C:\Users\Omar\Downloads\VisualData2project\scale-ai-templates-master\scale-ai-templates-master\apps\time-series-forecasting\archive\ProductPriceIndex.csv")

    # Filter data based on selected product
    veggDf['year'] = pd.to_datetime(veggDf['date']).dt.year
    # Convert 'price' column to numeric type, coerce non-numeric values to NaN
    veggDf = veggDf[veggDf[selected_price_type]!='$']
    veggDf[selected_price_type] = veggDf[selected_price_type].replace('[\$,]', '', regex=True).astype(float)

    # Drop rows with NaN values in 'price' column

    veggDf.dropna(subset=[selected_price_type], inplace=True)
    
    avg_prices = veggDf.groupby(['year', 'productname'])[selected_price_type].mean().reset_index()
    
    # Subset the dataframe with selected columns
    # price_data = avg_prices[selected_price_type]
    pivot_df = avg_prices.pivot(index='year', columns='productname', values=selected_price_type)

    # Calculate the correlation matrix
    correlation_matrix = pivot_df.corr()

    # Create the correlation heatmap plot
    fig = px.imshow(correlation_matrix, x=pivot_df.columns, y=pivot_df.columns,
                    labels=dict(x="", y="",color="Correlation"), color_continuous_scale="Viridis")

    fig.update_layout(title=f'Correlation Between {selected_price_type} of Different Crops',width=1800,  # Adjust the width of the plot
                      height=600)

    return fig

@app.callback(
    dash.dependencies.Output('average-price-plot', 'figure'),
    [dash.dependencies.Input('price-type-radio', 'value')]
)
def update_graph(price_type):

    veggDf = pd.read_csv(r"C:\Users\Omar\Downloads\VisualData2project\scale-ai-templates-master\scale-ai-templates-master\apps\time-series-forecasting\archive\ProductPriceIndex.csv")

    # Filter data based on selected product
    veggDf['year'] = pd.to_datetime(veggDf['date']).dt.year
    # Convert 'price' column to numeric type, coerce non-numeric values to NaN
    veggDf = veggDf[veggDf[price_type]!='$']
    veggDf[price_type] = veggDf[price_type].replace('[\$,]', '', regex=True).astype(float)

    # Drop rows with NaN values in 'price' column

    veggDf.dropna(subset=[price_type], inplace=True)
    
    vegetables = ['Romaine Lettuce', 'Red Leaf Lettuce', 'Potatoes', 'Iceberg Lettuce', 
                  'Green Leaf Lettuce', 'Celery', 'Cauliflower', 'Carrots', 'Broccoli Crowns', 
                  'Broccoli Bunches', 'Asparagus', 'Tomatoes']
    fruits = ['Strawberries', 'Oranges', 'Cantaloupe', 'Avocados', 'Flame Grapes', 
              'Thompson Grapes', 'Honeydews', 'Plums', 'Peaches', 'Nectarines']

    # Classify products as vegetables or fruits
    veggDf['product_type'] = np.where(veggDf['productname'].isin(vegetables), 'Vegetable', 
                                      np.where(veggDf['productname'].isin(fruits), 'Fruit', 'Other'))
    
    avg_prices = veggDf.groupby(['year', 'product_type'])[price_type].mean().reset_index()
    # Group by year and productname and calculate the average price
    # avg_prices = veggDf.groupby(['year', 'productname'])[price_type].mean().reset_index()
    
    
    # Create bar plot
    fig = px.bar(avg_prices, x='year', y=price_type, color='product_type',
                 labels={'year': 'Year', price_type: f'Average {price_type.capitalize()}'},
                 title=f'Average {price_type.capitalize()} of Vegetables vs Fruits per Year')
    return fig

# @app.callback(
#     dash.dependencies.Output('average-price-plot', 'figure'),
#     [dash.dependencies.Input('average-price-plot', 'id')]
# )
# def update_graph(_):
#     # Create bar plot of average prices
#     fig = px.bar(veggDf.groupby(['year', 'productname'])['price'].mean().reset_index(), x='year', y='price', color='productname', barmode='group',
#                  labels={'price': 'Average Price', 'year': 'Year', 'productname': 'Product Name'},
#                  title='Average Prices of Vegetables and Fruits per Year')
#     return fig

@app.callback(
    dash.dependencies.Output('Box-plot-graph', 'figure'),
    [dash.dependencies.Input('y-axis-checkbox', 'value')]
)
def update_graph(selected_values):
    # Generate box plot with selected y-axis variable
    fig = px.box(veggDf, x="productname", y=selected_values, color="productname")
    return fig


@app.callback(
    [Output("graph-forecast", "figure"), Output("graph-components", "figure")],
    [
        Input("radio-category", "value"),
        Input("radio-state", "value"),
        Input("radio-seasonality", "value"),
        Input("slider-forecast-days", "value"),
        Input("slider-uncertainty-samples", "value"),
        Input("range-years", "value"),
    ],
)
def run_forecast(category, state, seasonality, periods, uncertainty_samples, years):
    t0 = time.time()
    #col = f"('{category}', '{state}')"
    filtered_data = veggDf[(veggDf["productname"]==category) & (veggDf[state]!='$')]
    filtered_data[state] = filtered_data[state].replace('[\$,]', '', regex=True).astype(float)
    df = filtered_data[["date", state]].rename(columns={"date": "ds", state: "y"})

    m, forecast = run_prophet(
        df,
        weekly_seasonality=seasonality,
        periods=periods,
        uncertainty_samples=uncertainty_samples,
        start=years[0],
        end=years[1],
    )

    # Plot figures
    margin = dict(l=30, r=30, t=80, b=30)

    fig_forecast = plot_plotly(m, forecast).update_layout(
        title="Forecasting with Prophet",
        width=None,
        height=None,
        margin=margin,
    )

    fig_components = plot_components_plotly(m, forecast).update_layout(
        title="Seasonal Components", width=None, height=None, margin=margin
    )

    t1 = time.time()
    # print(f"Training and Inference time: {t1-t0:.2f}s.")

    return fig_forecast, fig_components


if __name__ == "__main__":
    app.run_server(debug=True)
