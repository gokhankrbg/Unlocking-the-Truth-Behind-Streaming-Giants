import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Unlocking the Truth Behind Streaming Giants ",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for background image and styling
# Mevcut CSS bloğunuzun içine bu yeni stilleri ekleyin


@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('MoviesOnStreamingPlatforms.csv')
    # Clean Rotten Tomatoes scores
    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(
        lambda x: float(x.split('/')[0]) if pd.notnull(x) and isinstance(x, str) and '/' in x else None
    )
    return df

def calculate_summary_stats(data):
    """Calculate summary statistics for a platform's data"""
    stats_dict = {
        'Total Movies': len(data),
        'Average RT Score': data['Rotten Tomatoes'].mean(),
        'Median RT Score': data['Rotten Tomatoes'].median(),
        'RT Score Std': data['Rotten Tomatoes'].std(),
        'Age Distribution': data['Age'].value_counts().to_dict()
    }
    return stats_dict

def perform_hypothesis_tests(netflix_data, disney_data):
    """Perform statistical hypothesis tests with proper handling of missing data"""
    # Remove rows with missing age ratings
    netflix_clean = netflix_data[netflix_data['Age'] != ''].copy()
    disney_clean = disney_data[disney_data['Age'] != ''].copy()

    # Age restriction test (Mann-Whitney U test)
    age_mapping = {'all': 0, '7+': 7, '13+': 13, '16+': 16, '18+': 18}
    netflix_ages = netflix_clean['Age'].map(age_mapping).dropna()
    disney_ages = disney_clean['Age'].map(age_mapping).dropna()

    age_stat, age_pval = stats.mannwhitneyu(
        disney_ages,
        netflix_ages,
        alternative='less'  # Testing if Disney+ has lower age restrictions
    )

    # RT scores test
    netflix_rt = netflix_data['Rotten Tomatoes'].dropna()
    disney_rt = disney_data['Rotten Tomatoes'].dropna()

    # Calculate effect size (Cohen's d)
    n1, n2 = len(netflix_rt), len(disney_rt)
    pooled_std = np.sqrt(
        ((n1 - 1) * netflix_rt.std() ** 2 + (n2 - 1) * disney_rt.std() ** 2) / (n1 + n2 - 2)
    )
    cohens_d = (disney_rt.mean() - netflix_rt.mean()) / pooled_std

    # Perform independent t-test
    scores_stat, scores_pval = stats.ttest_ind(netflix_rt, disney_rt)

    return {
        'age_test': {
            'statistic': age_stat,
            'p_value': age_pval,
            'n_netflix': len(netflix_ages),
            'n_disney': len(disney_ages)
        },
        'scores_test': {
            'statistic': scores_stat,
            'p_value': scores_pval,
            'effect_size': cohens_d,
            'n_netflix': n1,
            'n_disney': n2
        }
    }

def create_age_distribution_plot(netflix_data, disney_data):
    """Create age distribution comparison plot"""
    # Prepare data for plotting
    netflix_ages = netflix_data[netflix_data['Age'] != '']['Age'].value_counts()
    disney_ages = disney_data[disney_data['Age'] != '']['Age'].value_counts()

    # Create figure
    fig = go.Figure()

    # Add bars for each platform
    fig.add_trace(go.Bar(
        name='Netflix',
        x=netflix_ages.index,
        y=netflix_ages.values,
        text=netflix_ages.values,
        textposition='auto',
        marker_color='#E50914'
    ))

    fig.add_trace(go.Bar(
        name='Disney+',
        x=disney_ages.index,
        y=disney_ages.values,
        text=disney_ages.values,
        textposition='auto',
        marker_color='#113CCF'
    ))

    # Update layout
    fig.update_layout(
        title='Age Rating Distribution Comparison',
        xaxis_title='Age Rating',
        yaxis_title='Number of Movies',
        barmode='group',
        showlegend=True,
        height=500
    )

    return fig

def create_rt_scores_plot(netflix_data, disney_data):
    """Create Rotten Tomatoes scores comparison plot"""
    fig = go.Figure()

    # Add box plots for each platform
    fig.add_trace(go.Box(
        y=netflix_data['Rotten Tomatoes'].dropna(),
        name='Netflix',
        marker_color='#E50914',
        boxpoints='outliers'
    ))

    fig.add_trace(go.Box(
        y=disney_data['Rotten Tomatoes'].dropna(),
        name='Disney+',
        marker_color='#113CCF',
        boxpoints='outliers'
    ))

    # Update layout
    fig.update_layout(
        title='Rotten Tomatoes Score Distribution',
        yaxis_title='Rotten Tomatoes Score',
        showlegend=True,
        height=500
    )

    return fig

def create_yearly_trend_plot(netflix_data, disney_data):
    """Create yearly trend comparison plot with enhanced explanations"""
    # Calculate yearly averages and counts
    netflix_yearly = netflix_data.groupby('Year')['Rotten Tomatoes'].agg(['mean', 'count']).reset_index()
    disney_yearly = disney_data.groupby('Year')['Rotten Tomatoes'].agg(['mean', 'count']).reset_index()

    # Create figure
    fig = go.Figure()

    # Add lines for each platform
    fig.add_trace(go.Scatter(
        x=netflix_yearly['Year'],
        y=netflix_yearly['mean'],
        name='Netflix',
        mode='lines+markers',
        line=dict(color='#E50914'),
        hovertemplate="Year: %{x}<br>Average Score: %{y:.1f}<br>Movies: %{text}<extra></extra>",
        text=netflix_yearly['count']
    ))

    fig.add_trace(go.Scatter(
        x=disney_yearly['Year'],
        y=disney_yearly['mean'],
        name='Disney+',
        mode='lines+markers',
        line=dict(color='#113CCF'),
        hovertemplate="Year: %{x}<br>Average Score: %{y:.1f}<br>Movies: %{text}<extra></extra>",
        text=disney_yearly['count']
    ))

    # Update layout
    fig.update_layout(
        title='Average Rotten Tomatoes Scores by Year',
        xaxis_title='Year',
        yaxis_title='Average Score',
        showlegend=True,
        height=500
    )

    return fig

def main():
    # Title and introduction
    st.title("🎬 Unlocking the Truth Behind Streaming Giants")
    st.markdown("### Comparative Analysis of Age Restrictions and Movie Quality on Disney+ and Netflix: A Statistical Approach")
    st.markdown("##### prod.by Gökhan Karabağ")

    # Load data
    df = load_data()

    # Platform selection
    platforms = st.multiselect(
        "Select Platforms",
        options=['Netflix', 'Disney+', 'Hulu', 'Prime Video'],
        default=['Netflix', 'Disney+']
    )

    # Filter data based on selected platforms
    filtered_data = df[df[platforms].sum(axis=1) > 0]

    # Year range selector
    min_year = int(filtered_data['Year'].min())
    max_year = int(filtered_data['Year'].max())
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(2000, max_year)
    )

    # Filter data by selected years
    filtered_data = filtered_data[
        (filtered_data['Year'] >= year_range[0]) &
        (filtered_data['Year'] <= year_range[1])
    ]

    # Age restriction filter
    age_restrictions = st.multiselect(
        "Select Age Restrictions",
        options=['all', '7+', '13+', '16+', '18+'],
        default=['all', '7+', '13+', '16+', '18+']
    )

    # Filter data by selected age restrictions
    filtered_data = filtered_data[filtered_data['Age'].isin(age_restrictions)]

    # Calculate summary statistics
    netflix_data = filtered_data[filtered_data['Netflix'] == 1].copy()
    disney_data = filtered_data[filtered_data['Disney+'] == 1].copy()

    netflix_stats = calculate_summary_stats(netflix_data)
    disney_stats = calculate_summary_stats(disney_data)

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Descriptive Analysis",
        "🔍 Statistical Tests",
        "📈 Trend Analysis"
    ])

    with tab1:
        st.header("Descriptive Analysis")


        # Summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Netflix")
            st.metric("Total Movies", netflix_stats['Total Movies'])
            st.metric("Average RT Score", f"{netflix_stats['Average RT Score']:.2f}")
            st.metric("Median RT Score", f"{netflix_stats['Median RT Score']:.2f}")

            st.write("Age Distribution:")
            for age, count in netflix_stats['Age Distribution'].items():
                if age:  # Skip empty age ratings
                    st.write(f"- {age}: {count} movies ({count / netflix_stats['Total Movies'] * 100:.1f}%)")

        with col2:
            st.subheader("Disney+")
            st.metric("Total Movies", disney_stats['Total Movies'])
            st.metric("Average RT Score", f"{disney_stats['Average RT Score']:.2f}")
            st.metric("Median RT Score", f"{disney_stats['Median RT Score']:.2f}")

            st.write("Age Distribution:")
            for age, count in disney_stats['Age Distribution'].items():
                if age:  # Skip empty age ratings
                    st.write(f"- {age}: {count} movies ({count / disney_stats['Total Movies'] * 100:.1f}%)")

        # Visualizations
        st.plotly_chart(create_age_distribution_plot(netflix_data, disney_data), use_container_width=True)
        st.plotly_chart(create_rt_scores_plot(netflix_data, disney_data), use_container_width=True)

    with tab2:
        st.header("Statistical Tests")

        # Perform hypothesis tests
        test_results = perform_hypothesis_tests(netflix_data, disney_data)

        # Age restrictions test
        st.subheader("1. Age Restrictions Comparison")
        st.write("""
        **Research Question:** Is Disney+ primarily a platform for children's content compared to Netflix?

        **Hypothesis Test:**
        - H₀: Disney+ movies do not have lower age restrictions than Netflix movies.
        - H₁: Disney+ movies have lower age restrictions than Netflix movies.

        **Test Choice:**
        - The Mann-Whitney U test was chosen because the age restriction data is ordinal (e.g., 'all', '7+', '13+', etc.), and this test is suitable for comparing two independent samples with non-normal distributions.
        """)

        st.write(f"""
        **Test Results (Mann-Whitney U test):**
        - Sample sizes: Netflix (n={test_results['age_test']['n_netflix']}),
        Disney+ (n={test_results['age_test']['n_disney']})
        - Test Statistic: {test_results['age_test']['statistic']:.2f}
        - p-value: {test_results['age_test']['p_value']:.4f}
        """)

        if test_results['age_test']['p_value'] < 0.05:
            st.success(
                "**Conclusion:** There is statistical evidence that Disney+ movies have lower age restrictions than Netflix movies (p < 0.05). This supports the hypothesis that Disney+ is more focused on children's content."
            )
        else:
            st.info(
                "**Conclusion:** There is not enough statistical evidence to conclude that Disney+ movies have lower age restrictions. The data does not support the hypothesis that Disney+ is more focused on children's content."
            )

        # Quality comparison test
        st.subheader("2. Quality Comparison (Rotten Tomatoes Scores)")
        st.write("""
        **Research Question:** Are Netflix movies rated higher than Disney+ movies?

        **Hypothesis Test:**
        - H₀: There is no difference in Rotten Tomatoes scores between Netflix and Disney+ movies.
        - H₁: There is a difference in Rotten Tomatoes scores between the platforms.

        **Test Choice:**
        - An independent t-test was chosen because the Rotten Tomatoes scores are continuous and approximately normally distributed. The test compares the means of two independent groups.
        """)

        st.write(f"""
        **Test Results (Independent t-test):**
        - Sample sizes: Netflix (n={test_results['scores_test']['n_netflix']}),
        Disney+ (n={test_results['scores_test']['n_disney']})
        - Test Statistic: {test_results['scores_test']['statistic']:.2f}
        - p-value: {test_results['scores_test']['p_value']:.4f}
        - Effect size (Cohen's d): {test_results['scores_test']['effect_size']:.2f}
        """)

        if test_results['scores_test']['p_value'] < 0.05:
            st.success(
                "**Conclusion:** There is statistical evidence of a significant difference in movie ratings between platforms (p < 0.05). The effect size (Cohen's d) indicates the magnitude of this difference."
            )
        else:
            st.info(
                "**Conclusion:** There is not enough statistical evidence to conclude that ratings differ between platforms. The data does not support the hypothesis that one platform has higher-rated movies than the other."
            )

    with tab3:
        st.header("Trend Analysis")

        # Explanation of the trend analysis
        st.write("""
        **Trend Analysis Overview:**
        - This section analyzes the trends in movie ratings (Rotten Tomatoes scores) and the number of movies released each year on Netflix and Disney+.
        - The graph below shows the average Rotten Tomatoes scores and the number of movies for each year.
        - You can use the filters above to select specific platforms, years, and age restrictions.
        """)

        # Show trend plot
        st.plotly_chart(create_yearly_trend_plot(netflix_data, disney_data), use_container_width=True)

        # Add numeric summaries for selected period
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Netflix ({year_range[0]}-{year_range[1]})")
            filtered_stats = calculate_summary_stats(netflix_data)
            st.write(f"Number of Movies: {filtered_stats['Total Movies']}")
            st.write(f"Average RT Score: {filtered_stats['Average RT Score']:.2f}")
            st.write(f"Score Standard Deviation: {filtered_stats['RT Score Std']:.2f}")

        with col2:
            st.subheader(f"Disney+ ({year_range[0]}-{year_range[1]})")
            filtered_stats = calculate_summary_stats(disney_data)
            st.write(f"Number of Movies: {filtered_stats['Total Movies']}")
            st.write(f"Average RT Score: {filtered_stats['Average RT Score']:.2f}")
            st.write(f"Score Standard Deviation: {filtered_stats['RT Score Std']:.2f}")

        # Data quality note
        st.info("""
        **Note on Data Quality:**
        - Some movies may have missing age ratings or Rotten Tomatoes scores.
        - The analysis includes only movies with available data.
        - Trends might be affected by the different total number of movies on each platform.
        """)

if __name__ == '__main__':
    main()

