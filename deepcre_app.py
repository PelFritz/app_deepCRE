import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import os
from utils import prepare_seqs, compute_scores
from io import StringIO
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.title('DeepCRE')
st.subheader('Gene expression predictive models for plant species')
st.write("""
These models were trained to predict the probability of a gene from a plant species being highly expressed. The expect 
as inputs, the flanking regions of genes. We define flanking regions as 1500 nt upstream and 1500 nt downstream of the 
your gene of interest. sequences should look like the example
sequence GCCGTCCGBREAKGCGGCGCGT. Notice there is a word 'BREAK' in the sequence, this is not part of the normal DNA
sequence, but it tells us were your upstream sequence ends and were the downstream begins. The sequence in your fasta is\n
\>AT10g0560\n
GCCGTCCGBREAKGCGGCGCGT
""")
st.subheader('Model selection and file upload')
st.write("""Select a model to use for predictions. The SSR model is trained on Arabidopsis thaliana, while
the MSR model is trained on 4 species: A. thaliana, S. lycopersicum, S. bicolor and Z. mays.""")
model_type = st.selectbox('Model', options=['SSR', 'MSR'])
saved_models = {'SSR':'arabidopsis_model_1_promoter_terminator.h5', 'MSR': 'super_msr_model.h5'}
input_fasta = st.file_uploader(label="""
Upload a fasta file from your local machine containing you sequence of interest.
""")

if input_fasta is None:
    st.stop()

@st.cache_data
def run_model(data, selected_model):
    input_fasta = StringIO(data.getvalue().decode("utf-8"))
    onehot_seq, seq_ids = prepare_seqs(input_fasta)
    model = load_model(saved_models[selected_model])
    prediction = model.predict(onehot_seq).ravel()
    predicted_class = prediction > 0.5
    pred_df = pd.DataFrame({'gene_ids': seq_ids, 'predicted_prob': prediction, 'High expressed': predicted_class})
    scores = compute_scores(onehot_data=onehot_seq, keras_model=model)
    backend.clear_session()
    return scores, pred_df


saliency_scores, meta_df = run_model(input_fasta, model_type)
st.write(meta_df)

vis_group = st.selectbox(label="groupy_by", options=['High vs Low', 'gene_id'])
region = st.selectbox(label='Highlight region', options=['UTR', 'Promoter-Terminator'])

if vis_group == 'gene_id':
    rolling_window = st.selectbox(label='Choose rolling window', options=[10, 25, 50])
if vis_group == "High vs Low":
    for expressed_group, group_id in zip([1, 0], ['High', 'Low']):
        idxs = np.where(meta_df['High expressed'].values == expressed_group)[0]
        mean_group = np.mean(saliency_scores[idxs], axis=0)
        scores_df = pd.DataFrame({'nucleotide position': np.arange(mean_group.shape[0]),
                                  'A': mean_group[:, 0],
                                  'C': mean_group[:, 1],
                                  'G': mean_group[:, 2],
                                  'T': mean_group[:, 3]})
        chart_group = px.line(scores_df, x='nucleotide position', y=['A', 'C', 'G', 'T'],
                              color_discrete_map={'A': 'green', 'C': 'cornflowerblue', 'G': 'red', 'T': 'darkorange'},
                              title=f'Shap importance scores for gene predicted for {group_id} expression')
        if region == 'UTR':
            chart_group.add_vrect(x0=999, x1=999 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                  annotation_text="5'UTR", annotation_position="top left")
            chart_group.add_vrect(x0=1519, x1=1519 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                  annotation_text="3'UTR", annotation_position="top left")

        else:
            chart_group.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                  annotation_text="promoter", annotation_position="top left")
            chart_group.add_vrect(x0=1519 + 500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                  annotation_text="terminator", annotation_position="top left")

        chart_group.update_xaxes(ticktext=[-1000, -500, 'TSS', 'TTS', 500, 1000],
                                 tickvals=[0, 499, 999, 2019, 2519, 3019])
        chart_group.update_yaxes(tickformat=".4f")
        st.plotly_chart(chart_group)

else:
    st.write("""You can group genes together and compare group average saliency scores. This may be useful in 
    cases were you wanted to compare one gene family to another for example. Or functional comparisons.""")
    genes_g1 = st.multiselect(label="Group 1", options=meta_df['gene_ids'])
    genes_g2 = st.multiselect(label="Group 2", options=meta_df['gene_ids'])
    if len(genes_g1) != 0 and len(genes_g2) != 0:
        idx_genes_g1 = meta_df.index[meta_df['gene_ids'].isin(genes_g1)]
        idx_genes_g2 = meta_df.index[meta_df['gene_ids'].isin(genes_g2)]
        chart_group = make_subplots(rows=1, cols=2, subplot_titles=['Group 1', 'Group 2'], shared_yaxes=True)
        chart2_group = make_subplots(rows=1, cols=2, subplot_titles=['Group 1', 'Group 2'], shared_yaxes=True)
        for gene_idxs, group_id, col in zip([idx_genes_g1, idx_genes_g2], ['G1', 'G2'], [1, 2]):
            show_legend = True if group_id == 'G1' else False
            group_score_mean = np.mean(saliency_scores[gene_idxs], axis=0)
            max_score = group_score_mean.max()
            scores_df = pd.DataFrame({'nucleotide position': np.arange(group_score_mean.shape[0]),
                                      'A': group_score_mean[:, 0],
                                      'C': group_score_mean[:, 1],
                                      'G': group_score_mean[:, 2],
                                      'T': group_score_mean[:, 3]})
            chart_group.add_trace(go.Scatter(x=scores_df['nucleotide position'], y=scores_df['A'],
                                             marker=dict(color='green'), name='A',
                                             showlegend=show_legend, legendgroup='A'), row=1, col=col)
            chart_group.add_trace(go.Scatter(x=scores_df['nucleotide position'], y=scores_df['C'],
                                             marker=dict(color='cornflowerblue'), name='C',
                                             showlegend=show_legend, legendgroup='C'), row=1, col=col)
            chart_group.add_trace(go.Scatter(x=scores_df['nucleotide position'], y=scores_df['G'],
                                             marker=dict(color='red'), name='G',
                                             showlegend=show_legend, legendgroup='G'), row=1, col=col)
            chart_group.add_trace(go.Scatter(x=scores_df['nucleotide position'], y=scores_df['T'],
                                             marker=dict(color='darkorange'), name='T',
                                             showlegend=show_legend, legendgroup='T'), row=1, col=col)
            chart_group.update_xaxes(title_text="nucleotide position", row=1, col=col)
            chart_group.update_yaxes(title_text="saliency score", row=1, col=col)
            chart_group.update_xaxes(ticktext=[-1000, -500, 'TSS', 'TTS', 500, 1000],
                                     tickvals=[0, 499, 999, 2019, 2519, 3019])
            chart_group.update_yaxes(tickformat=".4f")

            if region == 'UTR':
                chart_group.add_vrect(x0=999, x1=999+500, fillcolor='grey', opacity=0.1, line_width=0,
                                      annotation_text="5'UTR", annotation_position="top left")
                chart_group.add_vrect(x0=1519, x1=1519+500, fillcolor='grey', opacity=0.1, line_width=0,
                                      annotation_text="3'UTR", annotation_position="top left")
            else:
                chart_group.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                      annotation_text="promoter", annotation_position="top left")
                chart_group.add_vrect(x0=1519+500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                      annotation_text="terminator", annotation_position="top left")

            # For chart 2
            ma = np.nan_to_num(pd.Series(np.mean(group_score_mean, axis=1)).rolling(rolling_window).sum().values)
            chart2_group.add_trace(go.Scatter(x=np.arange(len(ma)), y=ma, name=f'{group_id}',
                                              fill='tozeroy'), row=1, col=col)
            chart2_group.update_xaxes(title_text="nucleotide position", row=1, col=col)
            chart2_group.update_yaxes(title_text="saliency score", row=1, col=col)
            chart2_group.update_xaxes(ticktext=[-1000, -500, 'TSS', 'TTS', 500, 1000],
                                      tickvals=[0, 499, 999, 2019, 2519, 3019])
            chart2_group.update_yaxes(tickformat=".4f")

            if region == 'UTR':
                chart2_group.add_vrect(x0=999, x1=999+500, fillcolor='grey', opacity=0.1, line_width=0,
                                       annotation_text="5'UTR", annotation_position="top left")
                chart2_group.add_vrect(x0=1519, x1=1519+500, fillcolor='grey', opacity=0.1, line_width=0,
                                       annotation_text="3'UTR", annotation_position="top left")
            else:
                chart2_group.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                       annotation_text="promoter", annotation_position="top left")
                chart2_group.add_vrect(x0=1519+500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                       annotation_text="terminator", annotation_position="top left")


        chart_group.update_layout(title_text="Group saliency scores", showlegend=True)
        chart2_group.update_layout(title_text=f"Rolling sum of saliency score, window = {rolling_window}",
                                   showlegend=True)

        st.plotly_chart(chart_group)
        st.plotly_chart(chart2_group)
