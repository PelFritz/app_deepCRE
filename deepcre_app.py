import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import os
from utils import prepare_seqs, compute_scores
from io import StringIO
from plotly.subplots import make_subplots
import plotly.graph_objects as go
st.set_page_config(layout='wide')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.title(':red[DeepCRE]')
st.subheader('Gene expression predictive models for plant species')


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


tab1main, tab2main, tab3main = st.tabs(['Home', 'Predictions', 'Saliency Maps'])
with tab1main:
    st.write("""
    These models were trained to predict the probability of a gene from a plant species being highly expressed. They expect 
    as inputs, the flanking regions of genes. We define flanking regions as 1500 nt upstream and 1500 nt downstream of the 
    your gene of interest. The upstream sequence is obtained by anchoring at the gene start, taking 1000 nt before 
    and 500 nt after it. This gives you 1500 nt upstream. The downstream sequence is obtained by anchoring on the gene end
    taking 500 nt before and 1000 nt after it. This also gives you 1500 nt downstream.\n
    
    Sequences should look like the example sequence GCCGTCCGBREAKGCGGCGCGT. Notice there is a word 'BREAK' in the sequence,
    this is not part of the normal DNA sequence, but it tells us were your upstream sequence ends and were the downstream 
    begins. The sequence in your fasta is\n
    \>AT10g0560\n
    GCCGTCCGBREAKGCGGCGCGT
    """)
    col1, _ = st.columns(2)
    with col1:
        st.image('extraction.png', caption='How to extract flanking sequences')
    model_type = st.sidebar.selectbox("""Select a model to use for predictions. The SSR model is trained on Arabidopsis thaliana, while
    the Siamese MSR model is trained on 4 species: A. thaliana, S. lycopersicum, S. bicolor and Z. mays.""", options=['SSR', 'Siamese MSR'])
    saved_models = {'SSR': 'arabidopsis_model_1_promoter_terminator.h5', 'Siamese MSR': 'siamese_super_msr_model.h5'}
    input_fasta = st.sidebar.file_uploader(label="""Upload a fasta file""")

    if input_fasta is None:
        st.stop()

with tab2main:
    saliency_scores, meta_df = run_model(input_fasta, model_type)
    st.write(meta_df)

with tab3main:
    vis_group = st.sidebar.selectbox(label="groupy_by", options=['High vs Low', 'gene_id'])
    region = st.sidebar.selectbox(label='Highlight region', options=['UTR', 'Promoter-Terminator'])

    rolling_window = st.sidebar.selectbox(label='Choose rolling window', options=[10, 25, 50])
    if vis_group == "High vs Low":
        tab1_hl, tab2_hl, tab3_hl = st.tabs(['Line', 'Area', 'Histogram'])
        # High
        mean_high = np.mean(saliency_scores[np.where(meta_df['High expressed'].values == 1)[0]], axis=0)
        scores_high = pd.DataFrame({'nucleotide position': np.arange(mean_high.shape[0]),
                                    'A': mean_high[:, 0],
                                    'C': mean_high[:, 1],
                                    'G': mean_high[:, 2],
                                    'T': mean_high[:, 3]})
        # Low
        mean_low = np.mean(saliency_scores[np.where(meta_df['High expressed'].values == 0)[0]], axis=0)
        scores_low = pd.DataFrame({'nucleotide position': np.arange(mean_low.shape[0]),
                                   'A': mean_low[:, 0],
                                   'C': mean_low[:, 1],
                                   'G': mean_low[:, 2],
                                   'T': mean_low[:, 3]})
        with tab1_hl:
            chart_hl_line = make_subplots(rows=1, cols=2,
                                          subplot_titles=['Shap importance scores: prediction = High expression',
                                                          'Shap importance scores: prediction = Low expression'],
                                          shared_yaxes=True)
            for group_id, score_df, col in zip(['High', 'Low'], [scores_high, scores_low], [1, 2]):
                show_legend = True if group_id == 'High' else False
                chart_hl_line.add_trace(go.Scatter(x=score_df['nucleotide position'], y=score_df['A'],
                                                   marker=dict(color='green'), name='A',
                                                   showlegend=show_legend, legendgroup='A'), row=1, col=col)
                chart_hl_line.add_trace(go.Scatter(x=score_df['nucleotide position'], y=score_df['C'],
                                                   marker=dict(color='cornflowerblue'), name='C',
                                                   showlegend=show_legend, legendgroup='C'), row=1, col=col)
                chart_hl_line.add_trace(go.Scatter(x=score_df['nucleotide position'], y=score_df['G'],
                                                   marker=dict(color='red'), name='G',
                                                   showlegend=show_legend, legendgroup='G'), row=1, col=col)
                chart_hl_line.add_trace(go.Scatter(x=score_df['nucleotide position'], y=score_df['T'],
                                                   marker=dict(color='darkorange'), name='T',
                                                   showlegend=show_legend, legendgroup='T'), row=1, col=col)
                chart_hl_line.update_xaxes(title_text="nucleotide position", row=1, col=col)
                chart_hl_line.update_yaxes(title_text="saliency score", row=1, col=col)
                chart_hl_line.update_xaxes(ticktext=[-1000, -500, 'TSS', 'TTS', 500, 1000],
                                           tickvals=[0, 499, 999, 2019, 2519, 3019])
                chart_hl_line.update_yaxes(tickformat=".4f")
                if region == 'UTR':
                    chart_hl_line.add_vrect(x0=999, x1=999 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                            annotation_text="5'UTR", annotation_position="top left")
                    chart_hl_line.add_vrect(x0=1519, x1=1519 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                            annotation_text="3'UTR", annotation_position="top left")
                else:
                    chart_hl_line.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                            annotation_text="promoter", annotation_position="top left")
                    chart_hl_line.add_vrect(x0=1519 + 500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                            annotation_text="terminator", annotation_position="top left")
            st.plotly_chart(chart_hl_line, use_container_width=True)

        with tab2_hl:
            chart_hl_area = make_subplots(rows=1, cols=2,
                                          subplot_titles=[f'Rolling sum, window = {rolling_window}, prediction = High',
                                                          f'Rolling sum, window = {rolling_window}, prediction = Low'],
                                          shared_yaxes=True)
            for group_id, score_df, colname, col_idx in zip(['High', 'Low'], [scores_high, scores_low], st.columns(2),
                                                            [1, 2]):
                with colname:
                    ma = np.nan_to_num(pd.Series(np.mean(score_df[['A', 'C', 'G', 'T']], axis=1)).rolling(rolling_window).sum().values)
                    chart_hl_area.add_trace(go.Scatter(x=np.arange(len(ma)), y=ma, name=f'{group_id}', fill='tozeroy'),
                                            row=1, col=col_idx)
                    chart_hl_area.update_xaxes(title_text="nucleotide position", row=1, col=col_idx)
                    chart_hl_area.update_yaxes(title_text="saliency score", row=1, col=col_idx)
                    chart_hl_area.update_xaxes(ticktext=[-1000, -500, 'TSS', 'TTS', 500, 1000],
                                               tickvals=[0, 499, 999, 2019, 2519, 3019])
                    chart_hl_area.update_yaxes(tickformat=".4f")

                    if region == 'UTR':
                        chart_hl_area.add_vrect(x0=999, x1=999 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                                annotation_text="5'UTR", annotation_position="top left")
                        chart_hl_area.add_vrect(x0=1519, x1=1519 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                                annotation_text="3'UTR", annotation_position="top left")
                    else:
                        chart_hl_area.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                                annotation_text="promoter", annotation_position="top left")
                        chart_hl_area.add_vrect(x0=1519 + 500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                                annotation_text="terminator", annotation_position="top left")
            st.plotly_chart(chart_hl_area, use_container_width=True)

        with tab3_hl:
            chart_hl_hist = make_subplots(rows=1, cols=2, subplot_titles=['Histogram of saliency scores: prediction = High',
                                                                          'Histogram of saliency scores: prediction = Low'],
                                          shared_yaxes=True)
            for group_id, score_df, colname, col_idx in zip(['High', 'Low'], [scores_high, scores_low], st.columns(2),
                                                            [1, 2]):
                with colname:
                    chart_hl_hist.add_trace(go.Histogram(x=np.mean(score_df[['A', 'C', 'G', 'T']], axis=1),
                                                         name=f'{group_id}'), row=1, col=col_idx)
                    chart_hl_hist.update_xaxes(title_text="Bins", row=1, col=col_idx)
                    chart_hl_hist.update_yaxes(title_text="saliency score", row=1, col=col_idx)
                    chart_hl_hist.update_xaxes(tickformat=".4f")

            st.plotly_chart(chart_hl_hist, use_container_width=True)


    else:
        st.write("""You can group genes together and compare group average saliency scores. This may be useful in 
            cases were you wanted to compare one gene family to another for example. Or functional comparisons.""")
        tab1_gl, tab2_gl = st.tabs(['Line and Area', 'Line and Histogram'])
        genes_g1 = st.sidebar.multiselect(label="Group 1", options=meta_df['gene_ids'])
        genes_g2 = st.sidebar.multiselect(label="Group 2", options=meta_df['gene_ids'])

        with tab1_gl:
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

                st.plotly_chart(chart_group, use_container_width=True)
                st.plotly_chart(chart2_group, use_container_width=True)

        with tab2_gl:
            if len(genes_g1) != 0 and len(genes_g2) != 0:
                idx_genes_g1 = meta_df.index[meta_df['gene_ids'].isin(genes_g1)]
                idx_genes_g2 = meta_df.index[meta_df['gene_ids'].isin(genes_g2)]
                chart_group = make_subplots(rows=1, cols=2, subplot_titles=['Group 1', 'Group 2'], shared_yaxes=True)
                chart2_group = make_subplots(rows=1, cols=2, subplot_titles=['Group 1', 'Group 2'], shared_yaxes=True)
                for gene_idxs, group_id, col, colname in zip([idx_genes_g1, idx_genes_g2], ['G1', 'G2'], [1, 2],
                                                             st.columns(2)):
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
                        chart_group.add_vrect(x0=999, x1=999 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                              annotation_text="5'UTR", annotation_position="top left")
                        chart_group.add_vrect(x0=1519, x1=1519 + 500, fillcolor='grey', opacity=0.1, line_width=0,
                                              annotation_text="3'UTR", annotation_position="top left")
                    else:
                        chart_group.add_vrect(x0=0, x1=999, fillcolor='blue', opacity=0.05, line_width=0,
                                              annotation_text="promoter", annotation_position="top left")
                        chart_group.add_vrect(x0=1519 + 500, x1=3019, fillcolor='blue', opacity=0.05, line_width=0,
                                              annotation_text="terminator", annotation_position="top left")

                    # For chart 2
                    with colname:
                        chart2_group.add_trace(go.Histogram(x=np.mean(scores_df[['A', 'C', 'G', 'T']], axis=1),
                                                            name=f'{group_id}'), row=1, col=col)
                        chart2_group.update_xaxes(title_text="Bins", row=1, col=col)
                        chart2_group.update_yaxes(title_text="saliency score", row=1, col=col)
                        chart2_group.update_xaxes(tickformat=".4f")

                    chart2_group.update_layout(title_text=f"Histogram saliency scores",
                                               showlegend=True)

                st.plotly_chart(chart_group, use_container_width=True)
                st.plotly_chart(chart2_group, use_container_width=True)
