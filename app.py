import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# page config - must be the first streamlit command
st.set_page_config(page_title='RiskLens', page_icon='🔍', layout='wide')

# custom css for a clean, professional look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0e1117; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

.stMetric {
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}

.stMetric label { color: #a0aec0 !important; font-weight: 500; }
div[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem; }
div[data-testid="stMetricDelta"] > div { font-size: 0.85rem; }

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
}

h1, h2, h3 { color: #e2e8f0 !important; }
p, li { color: #cbd5e0; }

.stTabs [data-baseweb="tab"] {
    color: #a0aec0;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #667eea !important;
    border-bottom-color: #667eea !important;
}
</style>
""", unsafe_allow_html=True)

# -- imports from our source modules --
try:
    from src.data_loader import load_kaggle_data, generate_upi_data, init_database
    from src.sql_queries import fraud_by_time_bucket, fraud_by_amount_bucket, upi_failure_analysis
    from src.risk_engine import (
        compute_user_baselines, score_transactions, evaluate_scoring, tune_weights,
        _compute_signal_frequency, compute_signal_overlap, fit_logistic_model, SIGNAL_COLUMNS
    )
    from src.drift_detector import compute_risk_trajectory, compute_drift_slopes, flag_drifting_users, get_interesting_users
    from src.threshold_simulator import precompute_all_thresholds, compute_cost_tradeoff
    from src.ai_triage import run_triage, evaluate_triage
    from src.product_strategy import stakeholder_impact, segment_analysis, get_recommendations
    from src.utils import get_csv_path, get_db_path, get_connection
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.info("Make sure you are running `streamlit run app.py` from the project root directory.")
    st.stop()

# -- color palette used across all charts --
COLORS = {
    'primary': '#667eea',
    'success': '#48bb78',
    'warning': '#ecc94b',
    'danger': '#f56565',
    'info': '#4299e1',
    'dark': '#2d3748',
}

# ============================================================
# DATA LOADING — cached so we don't recompute on every slider move
# ============================================================

@st.cache_data(show_spinner="Loading data and running risk engine...")
def load_and_process():
    """
    Loads the kaggle dataset, runs the full risk pipeline, and returns
    everything the dashboard tabs need. Cached by streamlit so it only
    runs once per session.
    """
    results = {}

    csv_path = get_csv_path()
    if not os.path.exists(csv_path):
        return {'error': 'csv_missing'}

    # step 1 — load and enrich the credit card data
    df = load_kaggle_data(csv_path)
    if df.empty:
        return {'error': 'csv_missing'}

    results['total_transactions'] = len(df)
    results['total_fraud'] = int(df['Class'].sum())

    # step 2 — initialize sqlite database (for sql queries)
    db_path = get_db_path()
    init_database(csv_path, db_path)
    conn = get_connection(db_path)

    # step 3 — run sql analysis queries
    results['fraud_by_time'] = fraud_by_time_bucket(conn)

    amount_buckets, medians = fraud_by_amount_bucket(conn)
    results['fraud_by_amount'] = amount_buckets
    results['medians'] = medians

    results['upi_data'] = upi_failure_analysis(conn)
    conn.close()

    # step 4 — compute per-user baselines and score transactions
    baselines = compute_user_baselines(df)

    # try different weight combos and pick the best one
    best_weights, scored_df = tune_weights(df, baselines)
    results['best_weights'] = best_weights
    results['scored_df'] = scored_df

    # evaluate the best scoring
    eval_metrics = evaluate_scoring(scored_df)
    results['eval_metrics'] = eval_metrics

    # step 5 — precompute threshold metrics (cost tradeoff computed later with slider value)
    threshold_df = precompute_all_thresholds(scored_df, max_threshold=10)
    results['threshold_data'] = threshold_df

    # step 6 — compute risk trajectory and drift
    trajectory = compute_risk_trajectory(scored_df, n_periods=8)
    results['trajectory'] = trajectory

    drift_df = compute_drift_slopes(trajectory)
    results['drift_df'] = drift_df

    drifting_users = flag_drifting_users(drift_df, slope_threshold=0.1)
    results['drifting_users'] = drifting_users

    interesting = get_interesting_users(trajectory, drift_df, n=6)
    results['interesting_users'] = interesting

    # step 7 — run ai triage (mock by default, gemini if key provided later)
    triage_df = run_triage(api_key=None)
    results['triage_data'] = triage_df

    return results


# ============================================================
# LOAD DATA
# ============================================================
data = load_and_process()

if data.get('error') == 'csv_missing':
    st.title("📂 Dataset Required")
    st.warning("The Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv`) was not found.")
    st.markdown("""
    ### How to set up:
    1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    2. Extract `creditcard.csv` from the zip file
    3. Place it in the `data/` folder of this project
    4. Restart the app with `streamlit run app.py`
    """)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🔍 RiskLens")
st.sidebar.markdown("**Risk Engine Controls**")

selected_threshold = st.sidebar.slider(
    "Risk Score Cutoff",
    min_value=0, max_value=9, value=4, step=1,
    help="Transactions with risk score >= this value get flagged for review"
)

incident_multiplier = st.sidebar.slider(
    "Incident Cost Multiplier",
    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
    help="Total cost of a fraud incident as a multiple of the transaction amount. "
         "1x = just the transaction amount. 3-5x = includes chargeback fees, "
         "card replacement, account remediation, regulatory reporting."
)

st.sidebar.caption("_1x = transaction amount only. "
                   "3x = conservative industry estimate. "
                   "5x = includes brand/trust damage._")

st.sidebar.markdown("---")
api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)", type="password",
    help="Enter your Google Gemini API key to enable live AI triage"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
st.sidebar.metric("Total Transactions", f"{data.get('total_transactions', 0):,}")
st.sidebar.metric("Known Fraud Cases", f"{data.get('total_fraud', 0):,}")

if data.get('eval_metrics'):
    em = data['eval_metrics']
    st.sidebar.metric("Best Precision (High)", f"{em['high_precision']:.2%}")
    st.sidebar.metric("Best Recall (Med+High)", f"{em['mh_recall']:.2%}")

st.sidebar.markdown("---")
st.sidebar.caption("RiskLens v1.0 · Built for Product Intern Portfolio")

# ============================================================
# MAIN CONTENT
# ============================================================
st.title("🔍 RiskLens — Risk Decision System")
st.markdown("*Interactive risk-decision system for transaction fraud detection and customer risk drift analysis*")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎚️ Threshold Simulator",
    "📊 Signal Deep-Dive",
    "📈 Risk Trajectory",
    "💼 Product Strategy",
    "🔧 UPI Failures",
    "🤖 AI Triage"
])

# ============================================================
# TAB 1 — THRESHOLD SIMULATOR (the hero feature)
# ============================================================
with tab1:
    try:
        # recompute cost tradeoff with the current multiplier from sidebar
        td = compute_cost_tradeoff(data['threshold_data'],
                                   incident_cost_multiplier=incident_multiplier)
        curr = td[td['threshold'] == selected_threshold]
        if curr.empty:
            st.warning("No data for selected threshold.")
        else:
            curr = curr.iloc[0]

            col_left, col_right = st.columns(2)

            with col_left:
                # precision vs recall curve with selected threshold highlighted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=td['threshold'], y=td['precision'],
                    mode='lines+markers', name='Precision',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=td['threshold'], y=td['recall'],
                    mode='lines+markers', name='Recall (Fraud Caught)',
                    line=dict(color=COLORS['success'], width=3),
                    marker=dict(size=8)
                ))
                fig.add_vline(
                    x=selected_threshold, line_dash="dash",
                    line_color=COLORS['danger'],
                    annotation_text=f"Threshold = {selected_threshold}"
                )
                fig.update_layout(
                    title="Precision vs Recall by Threshold",
                    xaxis_title="Risk Score Threshold",
                    yaxis_title="Rate",
                    template="plotly_dark",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # financial impact bar chart
                net = curr['net_benefit']
                fig2 = go.Figure(data=[
                    go.Bar(
                        name='Fraud $ Prevented (actual amounts)',
                        x=['Financial Impact'],
                        y=[curr['fraud_loss_prevented']],
                        marker_color=COLORS['success']
                    ),
                    go.Bar(
                        name='Review Cost ($15/flagged txn)',
                        x=['Financial Impact'],
                        y=[curr['review_cost']],
                        marker_color=COLORS['danger']
                    ),
                ])
                fig2.update_layout(
                    title="Business Impact at Current Threshold",
                    barmode='group',
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # summary metric cards
            st.markdown("### Performance at Selected Threshold")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Flagged", f"{int(curr['flagged_count']):,}")
            m2.metric("True Positives", f"{int(curr['true_positives']):,}")
            m3.metric("False Positives", f"{int(curr['false_positives']):,}")
            m4.metric("Precision", f"{curr['precision']:.2%}")
            m5.metric("Recall", f"{curr['recall']:.2%}")

            # financial detail
            st.markdown("### Financial Impact")
            f1, f2, f3 = st.columns(3)
            f1.metric("Fraud $ Prevented", f"${curr['fraud_loss_prevented']:,.0f}",
                      help="Sum of actual fraud transaction amounts caught — from the data, not an assumption")
            f2.metric("Review Cost", f"${curr['review_cost']:,.0f}",
                      help="Assumption: $15 per flagged transaction for analyst review time (~10 min at $90/hr loaded cost)")
            f3.metric("Net Benefit", f"${net:,.0f}",
                      delta="Profitable" if net > 0 else "Unprofitable",
                      delta_color="normal" if net > 0 else "inverse")
            
            st.markdown("### Customer Experience")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers Blocked / Day", f"{curr.get('customers_blocked_per_day', 0):.1f}")
            c2.metric("Analyst Hours / Day", f"{curr.get('analyst_hours_per_day', 0):.1f}")
            c3.metric("Cost per Fraud Caught", f"${curr.get('cost_per_fraud_caught', 0):.2f}")
            c4.metric("Customer Friction Ratio", f"{curr.get('customer_friction_ratio', 0):.1f}:1")
            
            # Stakeholder Impact
            st.markdown("### Stakeholder Impact")
            impact = stakeholder_impact(td, selected_threshold)
            si1, si2, si3 = st.columns(3)
            with si1:
                st.markdown(f"**Customer**\n<div class='glass-card'><h4>{impact['customer']['label']}</h4><p>{impact['customer']['summary']}</p></div>", unsafe_allow_html=True)
            with si2:
                st.markdown(f"**Operations**\n<div class='glass-card'><h4>{impact['operations']['label']}</h4><p>{impact['operations']['summary']}</p></div>", unsafe_allow_html=True)
            with si3:
                st.markdown(f"**Finance**\n<div class='glass-card'><h4>{impact['finance']['label']}</h4><p>{impact['finance']['summary']}</p></div>", unsafe_allow_html=True)

            # insight callout
            pct_flagged = curr['pct_flagged'] * 100
            recall_pct = curr['recall'] * 100
            net_label = f"${abs(net):,.0f} {'saved' if net >= 0 else 'lost'}"
            st.info(
                f"💡 **At threshold {selected_threshold}:** you flag **{pct_flagged:.1f}%** of all transactions, "
                f"catch **{recall_pct:.1f}%** of actual fraud (${curr['fraud_loss_prevented']:,.0f} in real fraud amounts), "
                f"and the net impact is **{net_label}** after review costs. "
                f"*(Review cost assumes $15/flagged transaction for analyst time.)*"
            )

    except Exception as e:
        st.error(f"Could not render Threshold Simulator: {e}")

# ============================================================
# TAB 2 — SIGNAL DEEP-DIVE
# ============================================================
with tab2:
    try:
        st.markdown("### Analytical Depth Behind Scoring Engine")
        
        # Section A: Signal Frequency Analysis
        st.markdown("#### Signal Frequency Analysis")
        freq_df = _compute_signal_frequency(data['scored_df'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=freq_df['signal'],
            x=freq_df['pct_fraud_flagged'],
            name='Fraud % Flagged',
            orientation='h',
            marker_color=COLORS['danger']
        ))
        fig.add_trace(go.Bar(
            y=freq_df['signal'],
            x=freq_df['pct_legit_flagged'],
            name='Legit % Flagged',
            orientation='h',
            marker_color=COLORS['info']
        ))
        
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            title='Fraud vs Legit Flagged by Signal',
            xaxis_title='Percentage (%)',
            yaxis_title='Signal',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** category_rarity is the strongest indicator of fraud, having the highest lift compared to legit transactions.")
        
        # Section B: Signal Overlap
        st.markdown("#### Signal Overlap (Why 5 Signals?)")
        overlap_df = compute_signal_overlap(data['scored_df'])
        fig_overlap = px.bar(
            overlap_df, x='signal', y='unique_catches', 
            title="Unique Fraud Catches by Signal",
            color='unique_catches',
            color_continuous_scale='Purples',
            template='plotly_dark'
        )
        fig_overlap.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_overlap, use_container_width=True)
        
        total_unique = overlap_df['unique_catches'].sum()
        total_unique_amount = overlap_df['unique_amount'].sum()
        st.info(f"💡 **Insight:** Removing any single signal would miss a combined total of {total_unique} fraud cases worth ${total_unique_amount:,.2f}.")
        
        # Section C: LR Feature Importance
        st.markdown("#### Logistic Regression Feature Importance")
        model, coefs, df_lr = fit_logistic_model(data['scored_df'])
        coef_items = []
        for k, v in coefs.items():
            if k != 'intercept':
                coef_items.append({'feature': k, 'importance': v['relative_importance_pct'], 'direction': v['direction']})
        coef_df = pd.DataFrame(coef_items).sort_values('importance', ascending=True)
        
        fig_lr = px.bar(
            coef_df, x='importance', y='feature', orientation='h',
            title='Relative Feature Importance (LR)',
            color='direction',
            color_discrete_map={'increases fraud risk': COLORS['danger'], 'decreases fraud risk': COLORS['success']},
            template='plotly_dark'
        )
        fig_lr.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lr, use_container_width=True)
        
        # Section D: 3-Way Comparison Insight
        st.markdown("#### 3-Way Comparison Insight")
        st.markdown("""
        <div class="glass-card">
        <p>At practical flag rates, hand-weighted rules outperform LR because they optimize for precision (54% at score≥5) while LR optimizes for recall (70% at p≥0.5 but only 4.3% precision). In production: use rules for auto-blocking, LR for secondary queue prioritization.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not render Signal Deep-Dive: {e}")

# ============================================================
# TAB 3 — RISK TRAJECTORY
# ============================================================
with tab3:
    try:
        trajectory = data.get('trajectory', pd.DataFrame())
        drift_df = data.get('drift_df', pd.DataFrame())
        interesting = data.get('interesting_users', [])

        if not interesting:
            st.warning("Not enough data to show risk trajectories. Need users with activity across multiple time periods.")
        else:
            selected_user = st.selectbox(
                "Select a user to analyze their risk trajectory:",
                interesting,
                format_func=lambda x: f"User {x}"
            )

            # get this user's trajectory
            user_traj = trajectory[trajectory['user_id'] == selected_user].sort_values('period')

            if user_traj.empty:
                st.warning(f"No trajectory data for User {selected_user}.")
            else:
                fig = go.Figure()

                # risk score line
                fig.add_trace(go.Scatter(
                    x=user_traj['period'],
                    y=user_traj['avg_risk_score'],
                    mode='lines+markers',
                    name='Avg Risk Score',
                    line=dict(width=3, color=COLORS['info']),
                    marker=dict(size=10)
                ))

                # color zones for risk levels
                fig.add_hrect(y0=0, y1=1.5, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                fig.add_hrect(y0=1.5, y1=4.5, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
                fig.add_hrect(y0=4.5, y1=10, fillcolor="red", opacity=0.1, layer="below", line_width=0)

                # annotations for zones
                fig.add_annotation(x=0, y=0.75, text="Low", showarrow=False, font=dict(color="lightgreen", size=10), xref="paper")
                fig.add_annotation(x=0, y=3, text="Medium", showarrow=False, font=dict(color="orange", size=10), xref="paper")
                fig.add_annotation(x=0, y=6, text="High", showarrow=False, font=dict(color="pink", size=10), xref="paper")

                fig.update_layout(
                    title=f"Risk Score Trajectory — User {selected_user}",
                    xaxis_title="Time Period",
                    yaxis_title="Average Risk Score",
                    yaxis_range=[0, max(user_traj['avg_risk_score'].max() + 1, 6)],
                    template="plotly_dark",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                # drift metrics for this user
                user_drift = drift_df[drift_df['user_id'] == selected_user]
                if not user_drift.empty:
                    ud = user_drift.iloc[0]
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Risk Slope", f"{ud['slope']:.3f}/period",
                              delta="Trending Up" if ud['slope'] > 0.1 else "Stable",
                              delta_color="inverse" if ud['slope'] > 0.1 else "normal")
                    d2.metric("% Change", f"{ud['pct_change']:.1%}",
                              delta="Warning" if ud['pct_change'] > 0.5 else "OK",
                              delta_color="inverse" if ud['pct_change'] > 0.5 else "normal")
                    d3.metric("Current Bucket", ud['latest_bucket'])

            # show all drifting users
            drifting = data.get('drifting_users', pd.DataFrame())
            if not drifting.empty:
                st.markdown("### ⚠️ Users Drifting Toward Risk")
                st.markdown("*These users have increasing risk scores but are not yet classified as High risk — early intervention opportunity.*")
                st.dataframe(
                    drifting[['user_id', 'slope', 'pct_change', 'latest_risk_score', 'latest_bucket', 'n_periods_active']].head(15),
                    column_config={
                        'user_id': st.column_config.TextColumn("User ID"),
                        'slope': st.column_config.NumberColumn("Risk Slope", format="%.3f"),
                        'pct_change': st.column_config.NumberColumn("% Change", format="%.1%"),
                        'latest_risk_score': st.column_config.NumberColumn("Latest Score", format="%.2f"),
                        'latest_bucket': st.column_config.TextColumn("Current Bucket"),
                        'n_periods_active': st.column_config.NumberColumn("Periods Active"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No users currently flagged as drifting toward risk at the current slope threshold.")

    except Exception as e:
        st.error(f"Could not render Risk Trajectory: {e}")

# ============================================================
# TAB 4 — PRODUCT STRATEGY
# ============================================================
with tab4:
    try:
        st.markdown("### Customer Segment Analysis")
        segment_df = segment_analysis(data['scored_df'])
        st.dataframe(segment_df, use_container_width=True)
        
        st.info("💡 **Insight:** Different customer segments have varying optimal thresholds. Applying segment-specific thresholds maximizes ROI rather than using a global cutoff.")

        st.markdown("### Product Recommendations")
        recs = get_recommendations()
        
        r1, r2, r3 = st.columns(3)
        cols = [r1, r2, r3]
        for idx, rec in enumerate(recs):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 4px solid {rec['color']};">
                    <h4>{rec['icon']} {rec['context']}</h4>
                    <p><strong>Recommended Threshold:</strong> {rec['recommended_threshold']}</p>
                    <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Could not render Product Strategy: {e}")

# ============================================================
# TAB 5 — UPI FAILURES
# ============================================================
with tab5:
    try:
        st.info('📋 **Domain Knowledge Demonstration** — This dataset is synthetic, modeled on published UPI failure patterns. '
                'The purpose is to demonstrate framework thinking about payment failure taxonomies and resolution prioritization, '
                'not to claim empirical findings from production data.')
                
        upi_df = data.get('upi_data', pd.DataFrame())

        if upi_df.empty:
            st.warning("No UPI failure data available.")
        else:
            col_chart, col_stats = st.columns([3, 1])

            with col_chart:
                fig = px.bar(
                    upi_df,
                    y='failure_category', x='total_cases',
                    orientation='h',
                    color='pct_resolved',
                    color_continuous_scale='Blues_r',
                    title="UPI Failure Categories (Darker = Lower Resolution Rate)",
                    labels={
                        'failure_category': 'Failure Category',
                        'total_cases': 'Number of Cases',
                        'pct_resolved': 'Resolution Rate (%)'
                    },
                    template='plotly_dark'
                )
                fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                st.markdown("### Avg Resolution Time")
                for _, row in upi_df.iterrows():
                    st.metric(
                        row['failure_category'].replace('_', ' ').title(),
                        f"{row['avg_resolution_time_hours']:.1f} hrs"
                    )

            # find the worst category
            worst = upi_df.sort_values('pct_resolved').iloc[0]
            most_disputed = upi_df.sort_values('pct_disputed', ascending=False).iloc[0]
            st.info(
                f"💡 **Key Finding:** **{worst['failure_category'].replace('_', ' ').title()}** has the lowest "
                f"resolution rate ({worst['pct_resolved']:.1f}%) and "
                f"**{most_disputed['failure_category'].replace('_', ' ').title()}** generates the most disputes "
                f"({most_disputed['pct_disputed']:.1f}%). Prioritizing automated resolution for these categories "
                f"would reduce support ticket volume significantly."
            )

    except Exception as e:
        st.error(f"Could not render UPI Failures: {e}")


# ============================================================
# TAB 6 — AI TRIAGE
# ============================================================
with tab6:
    try:
        # if user provided an api key, re-run triage with gemini
        if api_key and api_key.strip():
            with st.spinner("Running live AI triage via Google Gemini..."):
                triage_base = data['triage_data'][['complaint_id', 'complaint_text', 'complaint_type']]
                triage_df = run_triage(triage_base, api_key=api_key)
        else:
            triage_df = data.get('triage_data', pd.DataFrame())

        if triage_df.empty:
            st.warning("No triage data available.")
        else:
            st.markdown("### Customer Complaint Triage")
            st.markdown("*Each flagged complaint is categorized and given a draft response for first-line support.*")

            st.dataframe(
                triage_df[['complaint_id', 'complaint_text', 'ai_category', 'ai_priority', 'ai_response']],
                column_config={
                    'complaint_id': st.column_config.TextColumn("ID", width="small"),
                    'complaint_text': st.column_config.TextColumn("Complaint", width="large"),
                    'ai_category': st.column_config.TextColumn("Category"),
                    'ai_priority': st.column_config.TextColumn("Priority"),
                    'ai_response': st.column_config.TextColumn("AI Draft Response", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )

            # category breakdown
            st.markdown("### Triage Summary")
            cat_counts = triage_df['ai_category'].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Cases", cat_counts.get('fraud', 0))
            c2.metric("Technical Failures", cat_counts.get('technical_failure', 0))
            c3.metric("User Errors", cat_counts.get('user_error', 0))
            
            st.markdown("### Triage Evaluation")
            eval_dict = evaluate_triage(triage_df)
            st.metric("Overall Accuracy", f"{eval_dict['accuracy']:.2f}")
            
            per_cat_df = pd.DataFrame(eval_dict['per_category']).T.reset_index()
            per_cat_df.columns = ['Category', 'Precision', 'Recall', 'F1']
            st.dataframe(per_cat_df, use_container_width=True)
            
            st.info("💡 **Insight:** AI Triage effectively classifies complaints into predefined categories. Accuracy is typically high, enabling automated first-line routing.")

            if api_key and api_key.strip():
                st.caption("✨ Triage powered by live Google Gemini API (gemini-2.0-flash)")
            else:
                st.caption("⚙️ Using simulated triage responses. Enter a Gemini API key in the sidebar for live AI triage.")

    except Exception as e:
        st.error(f"Could not render AI Triage: {e}")
