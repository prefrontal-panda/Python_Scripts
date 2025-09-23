#!/usr/bin/env python3

# This script is to correctly format the Question Level Data available from the VCAA Data Service website.
# In brief, we will take in the raw data, add a new column denoting the campus that the student is at, and finally, concatenating all tables together.

# Importing librarires required
import pandas as pd
import os as os
import argparse as argparse

# Making a function to dynamically add columns and concatenate
def mod_dataframe(year_level, input_folder):
    dataframes = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv') and f"Yr{year_level}_" in file_name:
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            # Clean column names - may have additional whitespace
            df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

            #print(f"\n→ Reading: {file_name}")
            #print(f"Columns: {df.columns.tolist()}")

            # Getting campus code
            campus = file_name.split('_')[2].split('.')[0]

            # Adding it to the dataframe
            df['Campus'] = campus

            # Making a full name column for student name
            df['Full Name'] = df[['First Name', 'Second Name', 'Surname']].fillna('').agg(
                lambda x: ' '.join(filter(None, x)), axis=1
                )

            # Removing Unnecesary Columns
            df.drop(columns=['First Name', 'Second Name', 'Surname'], inplace=True)

            # Renaming the student id column
            df.rename(columns={'School Student Id': 'Student ID'}, inplace=True)

            # Concatenating
            dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        print(f"No matching files found for Year {year_level} in {input_folder}.")
        return pd.DataFrame()
    
# Filling in blank student IDs
def fill_blank_id(df, masterlist_path):
    masterlist = pd.read_csv(masterlist_path)

    # Making sure leading and trailing whitespaces are removed
    masterlist['Full Name'] = masterlist['Full Name'].str.strip()

    # Making dictionary for lookup
    id_dictionary = dict(zip(masterlist['Full Name'], masterlist['Cases ID']))

    # List for unmatched students
    unmatched_students = []

    # Fill in missing IDs
    def id_fill(row):
        if pd.isna(row.get('Student ID')):
            matched_id = id_dictionary.get(row['Full Name'])
            if matched_id is None:
                unmatched_students.append(row['Full Name'])
            return matched_id
        return row['Student ID']

    df['Student ID'] = df.apply(id_fill, axis=1)

    # Identifying mising IDs
    unmatched_unique = sorted(set(unmatched_students)) # set drop duplicate student names
    if unmatched_unique:
        print("The following students were missing from the masterlist and still have no ID:")
        for name in unmatched_unique:
            print(f" - {name}")
    else:
        print("All missing student IDs were filled from the masterlist.")

    # Rearranging columns
    order = [
    'APS Year',
    'Reporting Test',
    'PSI',
    'Full Name',
    'Student ID',
    'Campus',
    'Home Group',
    'Outcome Name',
    'Dimension Name',
    'Testlet',
    'Testlet Order',
    'Testlet Question Order',
    'Question',
    'Student Response',
    'Student Score',
    'Correct Answer',
    'Max Score'] 

    df = df[order]

    return df

# Main function for cleaning
def main():
    parser = argparse.ArgumentParser(description='Cleaning StudentQuestionLevel files for NAPLAN.')
    parser.add_argument('--year', type=int, required=True, help='Year level (e.g. 3).')
    parser.add_argument('--input', type=str, required=True, help='Input folder with campus files.')
    parser.add_argument('--masterlist', type=str, required=True, help='Masterlist .csv with students full name and id.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the final file.')

    args = parser.parse_args()

    combined_df = mod_dataframe(args.year, args.input)
    if combined_df.empty:
        print("No data to process. Exiting.")
        return

    final_df = fill_blank_id(combined_df, args.masterlist)
    final_df.to_csv(args.output, index=False)
    print(f"Final combined file saved to: {args.output}")

if __name__ == '__main__':
    # Making sure files are in correct format
    print("Files should be named as: StudentQuestionLevel_Yr[x]_[Campus Code].csv")
    form_check = input("Do your files follow this format (y/n)?").strip().lower() # Ensuring lower case
    
    if form_check == "y":
        main()
    elif form_check == "n":
        print("Please ensure that your files follow the naming convention required.")
    else:
        print("Please type 'y' or 'n'.")