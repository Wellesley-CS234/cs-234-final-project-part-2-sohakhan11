import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# Page config
st.set_page_config(
    page_title="Wikipedia Attention to Economists",
    layout= "wide"
)


# Load data
def load_data():
    df = pd.read_csv("data/final_streamlit_dataset.csv")

    # Manipulate the date column to extract yyyy & mm
    df["date"] = df["date"].astype(str)
    df["year"] = df["date"].str[:4].astype(int)
    df["month"] = df["date"].str[4:6].astype(int)

    # Add the Nobel Alignment column
    df["years_from_nobel"] = np.where(
        df["is_nobel"] == True,
        df["year"] - df["nobel_year"],
        np.nan
    )

    return df
df = load_data()


# Sidebar filters
st.sidebar.header("Filter by Year")

year_range = st.sidebar.slider(
    "Year range",
    int(df["year"].min()),
    int(df["year"].max()),
    (2017, 2025)
)

nobel_filter = st.sidebar.selectbox(
    "Nobel status",
    ["All", "Nobel winners", "Non-Nobel"]
)

df_filtered = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

if nobel_filter == "Nobel winners":
    df_filtered = df_filtered[df_filtered["is_nobel"] == True]
elif nobel_filter == "Non-Nobel":
    df_filtered = df_filtered[df_filtered["is_nobel"] == False]


# Introduction
st.title("Wikipedia Attention Dynamics of Economists (2017–2025)")

st.markdown("""Hi everyone! Welcome to my final project.
            This page presents the final stage of my data analysis project, which 
            explores public attention to economists on Wikipedia using monthly pageview 
            data from 2017 to 2025. My research focuses on understanding how public 
            visibility changes over time and whether major academic recognition, specifically 
            the Nobel Prize in Economics, reshapes attention patterns.
""")

st.markdown("""Using Wikipedia pageviews as a proxy for public interest, I examine whether 
            Nobel-winning economists receive systematically higher attention and whether 
            this attention follows recognizable patterns, such as spikes around award years 
            followed by periods of decay. More broadly, I explore whether pageviews can reasonably 
            be interpreted as a measure of an economist's popularity or public visibility, and 
            where the limitations of such a metric lie. Throughout this page, I walk through the 
            different analyses I conducted, including attention dynamics over time, comparisons 
            between Nobel and non-Nobel economists, and modeling attempts to assess whether attention 
            alone can predict major academic recognition. Together, these analyses highlight both
            the insights and limitations of using large-scale online behavior data to study public 
            engagement with academic figures.""")


# Data Summary
st.header("Data Summary")
st.write("There are clear imbalances in the data that are important to acknowledge. The dataset"
"representing all economists includes individuals from across history, including economists from centuries"
"ago, after extensive cleaning and filtering. In contrast, the Nobel economist subset consists only of"
"economists who received the Nobel Prize between 2017 and 2025, making it a much smaller and more recent group.")

st.write("Additionally, many economists had to be excluded from the final dataset due to data availability "
"constraints, such as the absence of a Wikipedia article, missing Wikidata QIDs, or incomplete metadata."
"As a result, the analysis is necessarily limited to economists who are well-documented within Wikipedia and " \
"Wikidata, which introduces a visibility bias favoring more prominent or institutionally recognized figures."
"These imbalances do not invalidate the analysis, but they do shape how the results should be interpreted."
"Rather than comparing equivalent populations, the findings highlight differences in attention patterns within " \
"the available public record, reflecting how recognition, documentation, and historical prominence influence visibility on Wikipedia.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total economists", df["qid"].nunique())
with col2:
    st.metric("Nobel economists", df[df["is_nobel"] == True]["qid"].nunique())
with col3:
    st.metric("Total observations", len(df))

preview_data = st.toggle("Data Preview")
if preview_data:
    rows = st.slider("View rows", 0, 1000, 10)
    st.dataframe(df_filtered.head(rows))


# Overall Attention Over Time
st.header("Overall Attention Over Time")
st.markdown("This visualization shows how overall Wikipedia attention to "
            "economists has evolved between 2017 and 2025. Overall attention" 
            "remains relatively stable over time, with short-lived spikes "
            "driven by external events or recognition moments rather than "
            "long-term growth in public interest.")

total_views = (
    df_filtered
    .groupby("year", as_index=False)["views"]
    .sum()
)

fig_total = px.line(
    total_views,
    x="year",
    y="views",
    title="Total Wikipedia Pageviews to Economists Over Time"
)
st.plotly_chart(fig_total, use_container_width=True)


# Distribution of Attention (Box Plot)
st.header("Distribution of Monthly Attention")
st.write("The distribution of monthly pageviews is somewhat right-skewed, indicating" 
         "that most economists receive relatively low attention while a small number" 
         "attract disproportionately large visibility. This imbalance highlights"
         "that Wikipedia attention is concentrated among a limited set of highly visible.")

df_box = df_filtered.copy()
df_box["Nobel Status"] = np.where(
    df_box["is_nobel"], "Nobel", "Non-Nobel"
)

fig_box = px.box(
    df_box,
    x="Nobel Status",
    y="views",
    points="outliers",
    log_y=True,
    title="Distribution of Monthly Pageviews (Log Scale)"
)
st.plotly_chart(fig_box, use_container_width=True)


# Attention Dynamics Around Nobel Prize
st.header("Attention Dynamics Around the Nobel Prize")
st.write("This plot aligns Nobel-winning economists by the year" 
         "of their award to examine how attention changes relative"
         "to this announcement. A clear surge in attention around the Nobel"
         "year can be observed , followed by a gradual decline, suggesting"
           "that public interest responds strongly to major recognition but decays over time.")

nobel_df = df_filtered[
    (df_filtered["is_nobel"] == True) &
    (df_filtered["years_from_nobel"].between(-5, 5))
]

aligned = (
    nobel_df
    .groupby("years_from_nobel", as_index=False)["views"]
    .mean()
)

fig_align = px.line(
    aligned,
    x="years_from_nobel",
    y="views",
    markers=True,
    title="Average Attention Relative to Nobel Year"
)
fig_align.add_vline(x=0, line_dash="dash", annotation_text="Nobel Year")

st.plotly_chart(fig_align, use_container_width=True)


# Individual Nobel Attention Heatmap
st.header("Individual Nobel Attention Patterns")
st.write("Not surprisingly, the heatmap reveals substantial" 
        "heterogeneity in how individual Nobel laureates experience "
        "attention before and after their award.")

pivot = (
    nobel_df
    .pivot_table(
        index="name",
        columns="years_from_nobel",
        values="views",
        aggfunc="mean"
    )
)

fig_heat = px.imshow(
    pivot,
    aspect="auto",
    color_continuous_scale="Viridis",
    title="Heatmap of Attention Around Nobel Prize"
)

st.plotly_chart(fig_heat, use_container_width=True)


# Age vs Attention
st.header("Age and Visibility")
st.markdown("This analysis explores the relationship between "
            "economists' birth year and their average Wikipedia attention.")

age_df = (
    df_filtered
    .groupby(["qid", "name", "birth_year", "is_nobel"], as_index=False)
    .agg(avg_views=("views", "mean"))
)

fig_age = px.scatter(
    age_df,
    x="birth_year",
    y="avg_views",
    color="is_nobel",
    log_y=True,
    hover_name="name",
    title="Average Attention vs Birth Year"
)

st.plotly_chart(fig_age, use_container_width=True)


# Text Classification
st.header("Economist Classification by Field")

st.markdown("""
Using a zero-shot text classifier on Wikipedia summaries,
economists were categorized into common economic subfields.
""")

field_counts = (
    df.groupby("econ_field_pred")["qid"]
    .nunique()
    .sort_values(ascending=False)
)

# Visualization: Attention by field 
st.bar_chart(field_counts)

field_choice = st.selectbox(
    "Select economic field",
    sorted(df["econ_field_pred"].dropna().unique())
)

field_df = df[df["econ_field_pred"] == field_choice]

monthly = (
    field_df.groupby("date")["views"]
    .mean()
)

st.line_chart(monthly)

# Visualization: Dist by attention
st.subheader("Distribution of Monthly Attention")

sample = df.groupby("date")["views"].mean().reset_index()

fig = px.box(
    sample,
    y="views",
    points="outliers",
    title="Distribution of Monthly Pageviews"
)

st.plotly_chart(fig, use_container_width=True)


# Visualization: Nobel vs. non in a field
st.subheader("Distribution by Field and Nobel Status")
compare_df = (
    df.groupby(["econ_field_pred", "is_nobel"])["views"]
    .mean()
    .reset_index()
)
st.dataframe(compare_df)

fig = px.bar(
    compare_df,
    x="econ_field_pred",
    y="views",
    color="is_nobel",
    barmode="group",
    title="Average Monthly Wikipedia Attention by Field and Nobel Status",
    labels={
        "econ_field_pred": "Predicted Field of Economics",
        "views": "Average Monthly Pageviews",
        "is_nobel": "Nobel Winner"
    }
)

st.plotly_chart(fig, use_container_width=True)


# Nobel Prediction Reflection
st.header("Can Pageviews Predict Nobel Winners?")

st.markdown(""" I attempted to predict Nobel winners using pageview-based features. 
            While overall accuracy was high due to class imbalance, **ROC AUC was close
            to random**, showing that **Wikipedia attention alone is a weak predictor
            of Nobel outcomes**. This highlights the limits of popularity-based metrics when modeling
            academic recognition.""")

image_path = 'Images/Table.png'
st.image(
   image_path,
   caption='Prediction Model Results',
   width=400
)


# Ethics & Limitations
st.header("Limitations & Ethical Considerations")

st.markdown("""
- Wikipedia pageviews reflect **public visibility**, not scholarly quality  
- Media coverage and language bias influence attention  
- Nobel Prizes are rare and socially mediated events  
- Gender, geography, and institutional prestige likely affect visibility  

These results should be interpreted as **patterns of attention**, not
measures of intellectual contribution.
""")

st.success("Thank you for exploring!")
