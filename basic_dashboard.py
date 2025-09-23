# Import libraries
import pandas as pd
import plotly.express as plotly
from faicons import icon_svg
from htmltools import TagList
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget, output_widget

# Reactive file handler
@reactive.Calc
def uploaded_data():
    file = input.f()

    # Debugging in case file does not get read
    #print("File:", file)
    if file is None or len(file) == 0:
        print("No file uploaded")
        return pd.DataFrame() # Return empty data frame if nothing uploaded
  
    # Read uploaded file
    try:
        path = file[0]["datapath"]
        print(f"File path: {path}")  # Debug: verify path
        df = pd.read_csv(path, encoding='utf-8')
        print(f"Data loaded: shape {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()
    
@reactive.Effect
def debug_outcomes():
    df = uploaded_data()
    if not df.empty and "Outcome Name" in df.columns:
        print("Unique outcomes:", df["Outcome Name"].unique())


# Main title
ui.page_opts(title="NAPLAN Student Performance", fillable=True)

# Sidebar
with ui.sidebar(title="Settings"):
    # File Input Button
    ui.input_file(id="f",
                  label="Please upload your file:", 
                  multiple=False,
                  width=None, button_label='Browse...',
                  placeholder='No file selected',
                  capture=None)
    
    # Checkbox
    ui.input_checkbox_group(
        id = "outcome_dimension",
        label = "Outcome",
        choices = ["Numeracy", "Writing", "Spelling", "Grammar and Punctuation", "Reading"],
        selected=["Numeracy", "Writing", "Spelling", "Grammar and Punctuation", "Reading"]
        )

# Value Boxes
with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("users")):
        "Number of students"

        @render.text
        def count():
            return filtered_df().shape[0]
            #return f"{filtered_df().shape[0]:,}"

    with ui.value_box(showcase=icon_svg("bullseye")):
        "Average Score"

        @render.text
        def av_score():
            data=filtered_df()
            if data.empty or "Student Score" not in data.columns:
                return "N/A"
            avg = data["Student Score"].mean()
            return f"{avg:.2f}" if pd.notna(avg) else "N/A"
        
# Cards
with ui.layout_columns():
    # Data frame
    with ui.card(full_screen=True):
        ui.card_header("Student data")

        @render.data_frame
        def new_table():
            data = filtered_df()
            # Select columns
            cols = [
                "Full Name",
                "Student ID",
                "Campus",
                "Outcome Name",
                "Dimension Name",
                "Testlet",
                "Testlet Order",
                "Testlet Question Order",
                "Question",
                "Student Response",
                "Student Score",
                "Max Score"
            ]

            # Filter cols to only those present in data.columns
            existing_cols = [col for col in cols if col in data.columns]

            # If no columns found, return empty dataframe with no error
            if not existing_cols:
                return pd.DataFrame()
            
            return render.DataGrid(data[existing_cols], filters=True)
        
    # Table of average score for each outcome's dimension    
    with ui.card(full_screen=True):
        ui.card_header("Average Score per Dimension")

        @render.data_frame
        def dimension_summary():
            data = filtered_df()

            # Building table
            if data.empty:
                return pd.DataFrame(columns=["Dimension", "Responses", "Average Score"])
            
            # Finding summary
            summary = (
                data.groupby("Dimension Name")
                .agg(
                    Responses=("Student Score", "count"),
                    Avg_Score=("Student Score", "mean"))
                .reset_index()
                .rename(columns={
                "Dimension Name": "Dimension",
                "Avg_Score": "Average Score"})
                )
        
            summary["Average Score"] = summary["Average Score"].round(2)

            return render.DataGrid(summary)

# Plot score per question
with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Score Distribution per Question")
        output_widget("box_plot", width="100%", height="600px")

        @render_widget
        def box_plot():
            data = filtered_df()
            cols_req = ["Question", "Student Score", "Dimension Name"]

            # Building table
            if data.empty or not all(col in data.columns for col in cols_req):
                return None # No plot when no data
            
            # Make a copy to avoid SettingWithCopyWarning
            data = data.copy()

            # Ensure "Question" is string for categorical x-axis
            data["Question"] = data["Question"].astype(str)

            # Plot
            fig_box = plotly.box(
                data,
                x="Question",
                y="Student Score",
                color="Dimension Name",
                title="Spread of Scores by Question",
                points="all",  # Show individual points
            )

            fig_box.update_layout(
                xaxis_title="Question ID",
                yaxis_title="Score",
                xaxis_tickangle=45,
                height=600
            )

            return fig_box

# Reactive filtering
@reactive.calc
def filtered_df():
    data = uploaded_data()

    # Returns empty data frame if no data uploaded
    if data is None or "Outcome Name" not in data.columns:
        return pd.DataFrame()

    # Inserting case-insensitivity
    if input.outcome_dimension():
    # Normalize data values for comparison:
       normalized_outcomes = (
           data["Outcome Name"]
           .str.lower()
           .str.replace("&", "and")
           .str.strip()
           )
    # Normalising checkboxes   
       selected_outcomes = [x.lower() for x in input.outcome_dimension()]
    # Filter based on normalized values:       
       return data[normalized_outcomes.isin(selected_outcomes)]
    else:
        return pd.DataFrame()