# Import required libraries
import copy
import dash
import pandas as pd
import numpy as np
from pathlib import Path
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

import utils.figures as figs
import utils.custom_funcs as cf

# Get relative data folder
PATH = Path(__file__).parent
DATA_PATH = PATH.joinpath('data').resolve()

ABOUT_THE_PROJECT = """
Target and Market provides the best candidates for a marketing campaign for an investment tracker mobile application with a freemium model.
Once you provide the number of users you would like to target, this tool works by identifying free users who are most like premium users in terms of their in-app behaviour and investment portfolio.
Please select the number of users that make sense for your campaign and marketing budget.
"""

ABOUT_THE_SEGMENTS = """
Once the best candidates have been identified, a clustering algorithm segments them. This enables you to
maximize the return on marketing investment by tailoring your marketing campaign to each segment.
"""

# Get data
user_data = pd.read_csv(DATA_PATH/'user_predictions.csv', index_col=0)
df = user_data.sort_values(by=['prediction', 'holdings', 'avg_session'], ascending=False)
df = df.loc[df.isPro==0]

# App
external_scripts = [
    {
         'src': 'https://kit.fontawesome.com/23a91de90f.js',
         'crossorigin':'anonymous'
    }
]

app = dash.Dash(__name__,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                external_scripts=external_scripts
)
app.title = 'Target and Market'
server = app.server


# Layout
app.layout = html.Div(
    [
        dcc.Store(id='aggregate_data'),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),
        # Top title or navbar
        html.Div(
            [
	            html.Div(
	                [
		                html.Img(src='assets/logo.png', height=60, id='app_logo'),
		                html.Div(
		                	[
		                		html.H1('Target and Market', id='title'),
		                		html.H3('Optimize your campaign', id='tagline')
		                	],
		                	className='topnav-titles'
		                ),
		            ],
		            className='topnav-content'
	            ),
            ],
            id='title_bar',
            className='topnav'
        ),
        # Main container of dashboard
        html.Div(
            [
                # Top bar with total data set stats
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H2(f'{user_data.loc[user_data.isPro==0].shape[0]:,.0f}'), html.P('Free Users')],
                                    className='mini_container',
                                ),
                                html.Div(
                                    [html.H2(f'{user_data.isPro.sum():,.0f}'), html.P('Paid Users')],
                                    className='mini_container',
                                ),
                                html.Div(
                                    [html.H2(f'{user_data.isPro.mean():.1%}'), html.P('Percent Paid')],
                                    className='mini_container',
                                ),
                            ],
                            id='top-info-container',
                            className='row container-display',
                        ),
                    ],
                    id='top-column',
                    className='twelve columns',
                ),
                # Statisitcs of all users
                # html.Div(
                #     [
                #         html.Div(
                #             [
                #                 html.Div(
                #                     [html.H2(f'${user_data.holdings.mean():,.0f}'), html.P('Avg. Holdings')],
                #                     className='mini_container',
                #                 ),
                #                 html.Div(
                #                     [html.H2(f'{user_data.avg_session.mean():,.0f} s'), html.P('Avg. Session')],
                #                     className='mini_container',
                #                 ),
                #                 html.Div(
                #                     [html.H2(f'{user_data.active_days.mean():.0f}'), html.P('Avg. Active Days')],
                #                     className='mini_container',
                #                 ),
                #             ],
                #             id='top-info-container-2',
                #             className='row container-display',
                #         ),
                #     ],
                #     id='top-column-2',
                #     className='twelve columns',
                # ),
                # Tool explanation
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(ABOUT_THE_PROJECT)
                            ],
                            className='box notes'
                        )
                    ],
                    id='model-explanation',
                    className='twelve columns'
                ),
                # Slider control container
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    'Select number of users for the campaign:',
                                    id='slider_label',
                                    className='control_label'
                                ),
                                dcc.Slider(
                                    id='user_slider',
                                    min=2500,
                                    max=20000,
                                    step=2500,
                                    value=10000,
                                    marks={
                                        2500: '2,500',
                                        5000: '5,000',
                                        7500: '7,500',
                                        10000: '10,000',
                                        12500: '12,500',
                                        15000: '15,000',
                                        17500: '17,500',
                                        20000: '20,000',
                                    },
                                    className='dcc_control'
                                ),
                            ],
                            className='pretty_container'
                        )
                    ],
                    id='slider_control',
                    className='twelve columns'
                ),
                # Subtitle
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2('Characteristics of selected users')
                            ],
                            className='subtitle'
                        )
                    ],
                    className='twelve columns'
                ),
                # Characteristics of selected group of users
                html.Div(
                    [
                        # Left column - pie chart
                        html.Div(
                            [
                                html.Div(
                                	[dcc.Loading(
                                		id='sector-1a-loading',
                                		children=[
                                			dcc.Graph(id='location_pie_chart')	
                                		],
                                		type='circle'
                                	)],
                                    className='pretty_container',
                                ),
                            ],
                            className='four columns',
                            id='pie_chart_area',
                        ),
                        # Right column - responsive tank charts
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Loading(
                                        	id='sector-1b-loading',
                                        	children=[
                                        		html.Div([dcc.Graph(id='holdings_graph', className='tank_chart')]),
		                                        html.Div([dcc.Graph(id='avg_session_graph', className='tank_chart')]),
		                                        html.Div([dcc.Graph(id='active_days_graph', className='tank_chart')])
		                                    ],
		                                    type='circle'
	                                    ),
                                    ],
                                    id='chart_area',
                                    className='pretty_container',
                                ),
                            ],
                            id='right_column',
                            className='eight columns',
                        ),
                    ],
                    className='row flex-display',
                ),
                # Clustering explanation
                # html.Div(
                #     [
                #         html.Div(
                #             [
                #                 html.P(ABOUT_THE_SEGMENTS)
                #             ],
                #             className='box notes'
                #         )
                #     ],
                #     id='segments-explanation',
                #     className='twelve columns'
                # ),
                # Subtitle
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2('Segmentation of selected users')
                            ],
                            className='subtitle'
                        )
                    ],
                    className='twelve columns'
                ),
                # Results of clustering selected group
                html.Div(
                    [
                        # Left column - cluster details
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Loading(
                                            id='sector-2a-loading',
                                            children=[html.Div(
                                                [
                                                	html.H4('CLUSTER 1'),
                                                	html.P(id='cluster_one'),
                                                	html.Br(),
                                                	html.H4('CLUSTER 2'),
                                                	html.P(id='cluster_two'),
                                                	html.Br(),
                                                	html.H4('CLUSTER 3'),
                                                	html.P(id='cluster_three'),
                                                	html.Br(),
                                                	html.H4('CLUSTER 4'),
                                                	html.P(id='cluster_four'),
                                                	html.Br(),
                                                ],
                                            )],
                                            type='circle'
                                        )
                                    ],
                                    className='pretty_container',
                                ),
                            ],
                            className='four columns',
                            id='cluster_info_area',
                        ),
                        # Right column - Cluster chart
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Loading(
                                            id='sector-2b-loading',
                                            children=[html.Div(
                                                [dcc.Graph(id='clusters_graph')],
                                            )],
                                            type='circle'
                                        )
                                    ],
                                    id='chart_area_2',
                                    className='pretty_container',
                                ),
                            ],
                            id='right_column_2',
                            className='eight columns',
                        ),
                    ],
                    className='row flex-display',
                ),
            ],
            id='mainContainer'
        ),
        html.Div(
            # Footer
            [
                html.Div(
                	[
	                	html.P(
	                		[
	                			html.A(
	                				[html.I(className='fas fa-envelope fa-3x')],
	                				href='mailto:advait.iitb@gmail.com',
	                				target='_blank',
	                			),
	                			html.A(
	                				[html.I(className='fab fa-github fa-3x')],
	                				href='https://github.com/advaithasabnis',
	                				target='_blank',
	                			),
	                			html.A(
	                				[html.I(className='fab fa-linkedin fa-3x')],
	                				href='https://www.linkedin.com/in/advaithasabnis/',
	                				target='_blank',
	                			),
	                		],
	                	),
	                	html.Br(),
	                	html.H4(
		                    [
		                        html.I(className='far fa-copyright'),
		                        ' 2020 Advait Hasabnis'
		                    ],
	                	),
	                	html.Br(),
	               	],
	               	className='footer-content'
                )
            ],
            className='footer'
        )
    ],
    id='content'
)



# Clientside callback to resize container
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='resize'),
    Output('output-clientside', 'children'),
    [Input('holdings_graph', 'figure')],
)

# Slider -> Responsive Charts
@app.callback(
    [
        Output('location_pie_chart', 'figure'),
        Output('holdings_graph', 'figure'),
        Output('avg_session_graph', 'figure'),
        Output('active_days_graph', 'figure')
    ],
    [
        Input('user_slider', 'value'),
    ],
)
def update_graphs(user_slider):
    figure0 = figs.make_pie_chart(df, user_slider)
    figure1 = figs.make_holdings_figure(df, user_slider)
    figure2 = figs.make_avg_session_figure(df, user_slider)
    figure3 = figs.make_active_days_figure(df, user_slider)
    return figure0, figure1, figure2, figure3

# Slider -> Cluster Charts
@app.callback(
    [
        Output('clusters_graph', 'figure'),
        Output('cluster_one', 'children'),
        Output('cluster_two', 'children'),
        Output('cluster_three', 'children'),
        Output('cluster_four', 'children')
    ],
    [
        Input('user_slider', 'value'),
    ],
)
def update_clusters(user_slider):
    cluster_info, clusters, feature_range = cf.kmeans_cluster(df, user_slider)
    figure = figs.make_clusters_graph(clusters, feature_range)
    c1 = f'{cluster_info.loc[1, "size"]} users: characterized by a high average session time ({cluster_info.loc[1, "avg_session"]:.0f} secs) compared to the rest.'
    c2 = f'{cluster_info.loc[2, "size"]} users: characterized by a high number of active days ({cluster_info.loc[2, "active_days"]:.0f}) compared to the rest.'
    c3 = f'{cluster_info.loc[3, "size"]} users: characterized by a high value of investments (${cluster_info.loc[3, "holdings"]:,.0f}) compared to the rest.'
    c4 = f'{cluster_info.loc[4, "size"]} users: characterized by relatively low values of all three metrics.'
    return figure, c1, c2, c3, c4


# Main
if __name__ == '__main__':
    app.run_server(debug=True)