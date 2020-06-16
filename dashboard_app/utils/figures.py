import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go

pio.templates["verdana"] = go.layout.Template(
    layout_font=dict(family="verdana, arial", color="#ffffff"),
    layout_hoverlabel=dict(font_family="verdana, arial")
    )
pio.templates.default = "plotly_dark+verdana"


def make_pie_chart(df, user_slider):
    dff = df.iloc[:user_slider, :].copy()
    labels = dff.continent.value_counts().index
    values = dff.continent.value_counts().values
    colors = ['#290165', '#3e0297', '#5203c9', '#6704fb', '#8536fc', '#a468fd']
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        showlegend=False,
        marker=dict(colors=colors),
        ))
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    fig.update_layout(
        title=dict(text='Location', x=0.5, xanchor='center', font_size=16),
        margin=dict(l=0, b=0, t=25, r=0, pad=0),
        height=340,
        )
    return fig


def make_holdings_figure(df, user_slider):
    dff = df.iloc[:user_slider, :].copy()
    x_val = dff.holdings.mean()
    x_max = df.iloc[:5000, :].holdings.mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Average Holdings'],
        x=[x_val],
        orientation='h',
        hovertemplate='%{x:,.0f}<extra></extra>',
        width=1,
        showlegend=False,
        marker=dict(color='#6002EE'),
        ))
    fig.add_trace(go.Bar(
        y=['Average Holdings'],
        x=[x_max-x_val],
        orientation='h',
        hoverinfo='skip',
        width=1,
        marker=dict(color='#a3a3a3'),
        showlegend=False
        ))
    annotations = []
    annotations.append(dict(
        xref='x',
        xanchor='center',
        text=f'${x_val:,.0f}',
        font_size=16,
        x=x_val/2,
        y=0,
        showarrow=False
        ))

    fig.update_layout(
        title=dict(text='Average Holdings (USD)', font_size=16),
        annotations=annotations,
        barmode='stack',
        hovermode='closest',
        margin=dict(l=0, b=0, t=25, r=0),
        height=80
        )
    fig.update_yaxes(showticklabels=False, fixedrange=True)
    fig.update_xaxes(fixedrange=True, range=[0, x_max])

    return fig


def make_avg_session_figure(df, user_slider):
    dff = df.iloc[:user_slider, :].copy()
    x_val = dff.total_time.sum()/dff.sessions.sum()
    x_max = df.iloc[:5000, :].total_time.sum()/df.iloc[:5000, :].sessions.sum()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Average Session Time'],
        x=[x_val],
        orientation='h',
        hovertemplate='%{x:,.0f}<extra></extra>',
        width=1,
        showlegend=False,
        marker=dict(color='#6002EE'),
        ))
    fig.add_trace(go.Bar(
        y=['Average Session Time'],
        x=[x_max-x_val],
        orientation='h',
        hoverinfo='skip',
        width=1,
        marker=dict(color='#a3a3a3'),
        showlegend=False
        ))
    annotations = []
    annotations.append(dict(
        xref='x',
        xanchor='center',
        text=f'{x_val:,.0f}',
        font_size=16,
        x=x_val/2,
        y=0,
        showarrow=False
        ))

    fig.update_layout(
        title=dict(text='Average Session Time (sec)', font_size=16),
        annotations=annotations,
        barmode='stack',
        hovermode='closest',
        margin=dict(l=0, b=0, t=25, r=0),
        height=80
        )
    fig.update_yaxes(showticklabels=False, fixedrange=True)
    fig.update_xaxes(fixedrange=True, range=[0, x_max])

    return fig


def make_active_days_figure(df, user_slider):
    dff = df.iloc[:user_slider, :].copy()
    x_val = dff.active_days.mean()
    x_max = 31
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Average Active Days in May'],
        x=[x_val],
        orientation='h',
        hovertemplate='%{x:,.1f}<extra></extra>',
        width=1,
        showlegend=False,
        marker=dict(color='#6002EE'),
        ))
    fig.add_trace(go.Bar(
        y=['Average Active Days in May'],
        x=[x_max-x_val],
        orientation='h',
        hoverinfo='skip',
        width=1,
        marker=dict(color='#a3a3a3'),
        showlegend=False
        ))
    annotations = []
    annotations.append(dict(
        xref='x',
        xanchor='center',
        text=f'{x_val:,.1f}',
        font_size=16,
        x=x_val/2,
        y=0,
        showarrow=False
        ))

    fig.update_layout(
        title=dict(text='Average Active Days in May', font_size=16),
        annotations=annotations,
        barmode='stack',
        hovermode='closest',
        margin=dict(l=0, b=0, t=25, r=0),
        height=80
        )
    fig.update_yaxes(showticklabels=False, fixedrange=True)
    fig.update_xaxes(fixedrange=True, range=[0, x_max])

    return fig


def make_clusters_graph(clusters, feature_range):
    num_features = ['avg_session', 'active_days', 'holdings']
    fig = go.Figure()
    for index, row in clusters.iterrows():
        fig.add_trace(go.Scatter(
            x=['Avg. Session Time', 'Avg. Active Days', 'Avg. Holdings'],
            y=row[num_features],
            mode='markers+lines',
            marker=dict(size=16),
            line=dict(width=3),
            hoverinfo='skip',
            name=f'Cluster {index+1}',
            yaxis=f'y{index+1}'
            ))
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(
            domain=[0.2,1],
            showgrid=False,
            fixedrange=True
            ),
        yaxis=dict(
            title='Avg. Session Time (sec)',
            title_standoff=0,
            range=[-0.05,1.05],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=feature_range.avg_session.values,
            fixedrange=True
            ),
        yaxis2=dict(
            title='Avg. Active Days',
            title_standoff=0,
            anchor='free',
            overlaying='y',
            side='left',
            position=0.05,
            range=[-0.05,1.05],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=feature_range.active_days.values,
            fixedrange=True
            ),
        yaxis3=dict(
            title='Avg. Holdings ($)',
            title_standoff=10,
            anchor='x',
            overlaying='y',
            side='right',
            range=[-0.05,1.05],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            separatethousands=True,
            ticktext=feature_range.holdings.values,
            fixedrange=True
            ),
        yaxis4=dict(
            overlaying='y',
            range=[-0.05,1.05],
            visible=False,
            fixedrange=True
            ),
        margin=dict(l=0, t=0, b=0, r=0),
        height=410
        )
    fig.update_layout(
        legend=dict(
                orientation='h',
                x=0.6,
                y=1.1,
                xanchor='center'
            )
        )
    
    return fig