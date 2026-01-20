import os
import pandas as pd
from datetime import datetime
import warnings
import shutil
import tempfile
import re
from flask import Flask, request, render_template, redirect, url_for, send_file, flash, session
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

# --- Vercel Specific Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, '..', 'templates')
static_dir = os.path.join(BASE_DIR, '..', 'static')

# Initialize Flask app
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_for_local_dev_only')

# --- Global Variables ---
CONSOLIDATED_OUTPUT_COLUMNS = [
    'Barcode', 'Processor', 'Channel', 'Category', 'Company code', 'Region',
    'Vendor number', 'Vendor Name', 'Status', 'Received Date', 'Re-Open Date',
    'Allocation Date', 'Clarification Date', 'Completion Date', 'Requester',
    'Remarks', 'Aging', 'Today'
]

# --- Helper Functions ---

def format_date_to_mdyyyy(date_series):
    """
    Formats a pandas Series of dates to MM/DD/YYYY string format.
    Handles potential mixed types and NaT values.
    """
    datetime_series = pd.to_datetime(date_series, errors='coerce')
    formatted_series = datetime_series.apply(
        lambda x: f"{x.month}/{x.day}/{x.year}" if pd.notna(x) else ''
    )
    return formatted_series

def clean_column_names(df):
    """
    Cleans DataFrame column names by:
    1. Lowercasing all characters.
    2. Replacing spaces with underscores.
    3. Removing special characters (keeping only alphanumeric and underscores).
    4. Removing leading/trailing underscores.
    """
    new_columns = []
    for col in df.columns:
        col = str(col).strip().lower()
        col = re.sub(r'\s+', '_', col)
        col = re.sub(r'[^a-z0-9_]', '', col)
        col = col.strip('_')
        new_columns.append(col)
    df.columns = new_columns
    return df

def get_workon_column(df_columns, target_keywords):
    """
    Manually finds the best matching *cleaned* column name from Workon P71 based on target keywords.
    Ignores case and considers the initial word of the column name.
    Returns the *cleaned* column name if found, otherwise None.
    This function expects df_columns to be already cleaned.
    """
    if not isinstance(target_keywords, list):
        target_keywords = [target_keywords]

    for keyword in target_keywords:
        cleaned_keyword_part = re.sub(r'[^a-z0-9_]', '', keyword.strip().lower()).strip('_')
        for df_col_cleaned in df_columns:
            if df_col_cleaned == cleaned_keyword_part:
                return df_col_cleaned
            if df_col_cleaned.startswith(cleaned_keyword_part):
                return df_col_cleaned
            if cleaned_keyword_part in df_col_cleaned: # More flexible but could be risky
                return df_col_cleaned
    return None


def consolidate_data_process(df_pisa, df_esm, df_pm7, df_workon, main_consolidated_output_file_path):
    """
    Reads PISA, ESM, PM7, and Workon P71 data, filters PISA, consolidates data.
    Returns df_consolidated_main (PISA/ESM/PM7) and df_consolidated_workon.
    """
    print("Starting data consolidation process...")
    print("All input DataFrames loaded successfully!")

    df_pisa = clean_column_names(df_pisa.copy())
    df_esm = clean_column_names(df_esm.copy())
    df_pm7 = clean_column_names(df_pm7.copy())
    df_workon = clean_column_names(df_workon.copy()) # Clean Workon P71 columns

    main_consolidated_rows = []
    workon_consolidated_rows = [] # Separate list for Workon rows
    today_date = datetime.now()
    today_date_formatted = today_date.strftime("%m/%d/%Y")


    # --- PISA Processing ---
    allowed_pisa_users = ["Goswami Sonali", "Patil Jayapal Gowd", "Ranganath Chilamakuri","Sridhar Divya","Sunitha S","Varunkumar N"]
    if 'assigned_user' in df_pisa.columns:
        df_pisa_filtered = df_pisa[df_pisa['assigned_user'].isin(allowed_pisa_users)].copy()
    else:
        df_pisa_filtered = df_pisa.copy()

    if 'barcode' not in df_pisa_filtered.columns:
        print("Error: 'barcode' column not found in PISA file (after cleaning). Skipping PISA processing.")
    else:
        df_pisa_filtered['barcode'] = df_pisa_filtered['barcode'].astype(str)
        for index, row in df_pisa_filtered.iterrows():
            new_row = {
                'Barcode': row['barcode'],
                'Company code': row.get('company_code'),
                'Vendor number': row.get('vendor_number'),
                'Received Date': row.get('received_date'),
                'Completion Date': None, 'Status': None , 'Today': today_date, 'Channel': 'PISA',
                'Vendor Name': row.get('vendor_name'),
                'Re-Open Date': None, 'Allocation Date': None,
                'Requester': None, 'Clarification Date': None, 'Aging': None, 'Remarks': None,
                'Region': None,
                'Processor': None, 'Category': None
            }
            main_consolidated_rows.append(new_row)
        print(f"Collected {len(df_pisa_filtered)} rows from PISA.")

    # --- ESM Processing ---
    if 'barcode' not in df_esm.columns:
        print("Error: 'barcode' column not found in ESM file (after cleaning). Skipping ESM processing.")
    else:
        df_esm['barcode'] = df_esm['barcode'].astype(str)
        for index, row in df_esm.iterrows():
            new_row = {
                'Barcode': row['barcode'],
                'Received Date': row.get('received_date'),
                'Status': row.get('state'),
                'Requester': row.get('opened_by'),
                'Completion Date': row.get('closed') if pd.notna(row.get('closed')) else None,
                'Re-Open Date': row.get('updated') if (row.get('state') or '').lower() == 'reopened' else None,
                'Today': today_date, 'Remarks': row.get('short_description'),
                'Channel': 'ESM',
                'Company code': None,'Vendor Name': None,
                'Vendor number': None, 'Allocation Date': None,
                'Clarification Date': None, 'Aging': None,
                'Region': None,
                'Processor': None,
                'Category': None
            }
            main_consolidated_rows.append(new_row)
        print(f"Collected {len(df_esm)} rows from ESM.")

    # --- PM7 Processing ---
    if 'barcode' not in df_pm7.columns:
        print("Error: 'barcode' column not found in PM7 file (after cleaning). Skipping PM7 processing.")
    else:
        df_pm7['barcode'] = df_pm7['barcode'].astype(str)

        for index, row in df_pm7.iterrows():
            new_row = {
                'Barcode': row['barcode'],
                'Vendor Name': row.get('vendor_name'),
                'Vendor number': row.get('vendor_number'),
                'Received Date': row.get('received_date'),
                'Status': row.get('task'),
                'Today': today_date,
                'Channel': 'PM7',
                'Company code': row.get('company_code'),
                'Re-Open Date': None,
                'Allocation Date': None, 'Completion Date': None, 'Requester': None,
                'Clarification Date': None, 'Aging': None, 'Remarks': None,
                'Region': None,
                'Processor': None, 'Category': None
            }
            main_consolidated_rows.append(new_row)
        print(f"Collected {len(df_pm7)} rows from PM7.")

    # --- Workon P71 Processing (Manual Mapping) ---
    # Find cleaned column names dynamically using get_workon_column
    workon_barcode_col = get_workon_column(df_workon.columns, ['key'])
    workon_category_col = get_workon_column(df_workon.columns, ['action'])
    workon_company_code_col = get_workon_column(df_workon.columns, ['Company_code', 'companycode'])
    workon_region_col = get_workon_column(df_workon.columns, ['country'])
    workon_vendor_number_col = get_workon_column(df_workon.columns, ['vendor_number', 'vendornumber','vendor_no.(header)'])
    workon_vendor_name_col = get_workon_column(df_workon.columns, ['name1_'])
    workon_status_col = get_workon_column(df_workon.columns, ['status'])
    workon_received_date_col = get_workon_column(df_workon.columns, ['updated'])
    workon_requester_col = get_workon_column(df_workon.columns, ['applicant'])
    workon_remarks_col = get_workon_column(df_workon.columns, ['summary'])

    # List of actually found columns for Workon
    found_workon_cols = {
        'Barcode': workon_barcode_col, 'Category': workon_category_col,
        'Company code': workon_company_code_col, 'Region': workon_region_col,
        'Vendor number': workon_vendor_number_col, 'Vendor Name': workon_vendor_name_col,
        'Status': workon_status_col, 'Received Date': workon_received_date_col,
        'Requester': workon_requester_col, 'Remarks': workon_remarks_col
    }
    
    # Check if all critical Workon columns were found (those explicitly mapped)
    critical_workon_cols_present = all(col_name is not None for col_name in found_workon_cols.values())

    if not critical_workon_cols_present:
        missing_cols_names = [k for k, v in found_workon_cols.items() if v is None]
        print(f"Warning: Missing critical Workon P71 columns for processing: {missing_cols_names}. Skipping Workon P71 processing.")
    else:
        for index, row in df_workon.iterrows():
            new_row = {
                'Barcode': row.get(found_workon_cols['Key']),
                'Processor': 'Jayapal',
                'Channel': 'Workon',
                'Category': row.get(found_workon_cols['Action']),
                'Company code': row.get(found_workon_cols['Company code']),
                'Region': row.get(found_workon_cols['Country']),
                'Vendor number': row.get(found_workon_cols['Vendor No. (Header)']),
                'Vendor Name': row.get(found_workon_cols['Name 1']),
                'Status': row.get(found_workon_cols['Status']),
                'Received Date': row.get(found_workon_cols['Updated']),
                'Re-Open Date': None,
                'Allocation Date': today_date_formatted,
                'Clarification Date': None,
                'Completion Date': None,
                'Requester': row.get(found_workon_cols['Applicant']),
                'Remarks': row.get(found_workon_cols['Summary']),
                'Aging': None,
                'Today': today_date
            }
            workon_consolidated_rows.append(new_row)
        print(f"Collected {len(df_workon)} rows from Workon P71.")

    # Process main consolidated data (PISA/ESM/PM7)
    df_consolidated_main = pd.DataFrame(main_consolidated_rows)
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_consolidated_main.columns:
            df_consolidated_main[col] = None
    df_consolidated_main = df_consolidated_main[CONSOLIDATED_OUTPUT_COLUMNS]

    # Process Workon consolidated data
    df_consolidated_workon = pd.DataFrame(workon_consolidated_rows)
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_consolidated_workon.columns:
            df_consolidated_workon[col] = None
    df_consolidated_workon = df_consolidated_workon[CONSOLIDATED_OUTPUT_COLUMNS]

    # Convert date columns to datetime objects for internal consistency BEFORE saving/further processing
    for df in [df_consolidated_main, df_consolidated_workon]:
        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col in ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('')
                elif col in ['Barcode', 'Company code', 'Vendor number']:
                    df[col] = df[col].astype(str).replace('nan', '')

    # Save a temporary consolidated file for PISA/ESM/PM7 data (for potential debugging/intermediate view)
    try:
        df_consolidated_main_for_save = df_consolidated_main.copy()
        for col in ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']:
            if col in df_consolidated_main_for_save.columns:
                df_consolidated_main_for_save[col] = format_date_to_mdyyyy(df_consolidated_main_for_save[col])
        df_consolidated_main_for_save.to_excel(main_consolidated_output_file_path, index=False)
        print(f"Main consolidated file (PISA/ESM/PM7) saved to: {main_consolidated_output_file_path}")
    except Exception as e:
        return False, f"Error saving main consolidated file: {e}"

    print("--- Consolidated Data Process Complete ---")
    return True, (df_consolidated_main, df_consolidated_workon)

def process_central_file_step2_update_existing(consolidated_main_df, central_file_input_path):
    """
    Step 2: Updates status of *existing* central file records based on consolidated_main data (PISA/ESM/PM7).
    """
    print(f"\n--- Starting Central File Status Processing (Step 2: Update Existing Barcodes) ---")

    try:
        converters = {'Barcode': str, 'Vendor number': str, 'Company code': str}
        df_central = pd.read_excel(central_file_input_path, converters=converters, keep_default_na=False)
        df_central_cleaned = clean_column_names(df_central.copy())

        for col in ['received_date', 're_open_date', 'allocation_date', 'completion_date', 'clarification_date', 'today']:
            if col in df_central_cleaned.columns:
                df_central_cleaned[col] = pd.to_datetime(df_central_cleaned[col], errors='coerce')

    except Exception as e:
        return False, f"Error loading Main Consolidated (DF) or Central (file) for processing (Step 2): {e}"

    if 'Barcode' not in consolidated_main_df.columns:
        return False, "Error: 'Barcode' column not found in the main consolidated file. Cannot proceed with central file processing (Step 2)."
    if 'barcode' not in df_central_cleaned.columns or 'status' not in df_central_cleaned.columns:
        return False, "Error: 'barcode' or 'status' column not found in the central file after cleaning. Cannot update status (Step 2)."

    consolidated_main_df['Barcode'] = consolidated_main_df['Barcode'].astype(str)
    df_central_cleaned['barcode'] = df_central_cleaned['barcode'].astype(str)

    df_central_cleaned['Barcode_compare'] = df_central_cleaned['barcode']

    consolidated_main_barcodes_set = set(consolidated_main_df['Barcode'].unique())
    print(f"Found {len(consolidated_main_barcodes_set)} unique barcodes in the main consolidated file for Step 2.")

    def transform_status_if_barcode_exists(row):
        central_barcode = str(row['Barcode_compare'])
        original_central_status = row['status']

        if central_barcode in consolidated_main_barcodes_set:
            if pd.isna(original_central_status) or \
               (isinstance(original_central_status, str) and original_central_status.strip().lower() in ['', 'n/a', 'na', 'none']):
                return original_central_status

            status_str = str(original_central_status).strip().lower()
            if status_str == 'new':
                return 'Untouched'
            elif status_str == 'completed':
                return 'Reopen'
            elif status_str == 'n/a':
                return 'New'
            else:
                return original_central_status
        else:
            return original_central_status

    df_central_cleaned['status'] = df_central_cleaned.apply(transform_status_if_barcode_exists, axis=1)
    df_central_cleaned = df_central_cleaned.drop(columns=['Barcode_compare'])

    print(f"Updated 'status' column in central file for Step 2 for {len(df_central_cleaned)} records.")

    try:
        common_cols_map = {
            'barcode': 'Barcode', 'channel': 'Channel', 'company_code': 'Company code',
            'vendor_name': 'Vendor Name', 'vendor_number': 'Vendor number',
            'received_date': 'Received Date', 're_open_date': 'Re-Open Date',
            'allocation_date': 'Allocation Date', 'completion_date': 'Completion Date',
            'requester': 'Requester', 'clarification_date': 'Clarification Date',
            'aging': 'Aging', 'today': 'Today', 'status': 'Status', 'remarks': 'Remarks',
            'region': 'Region', 'processor': 'Processor', 'category': 'Category'
        }

        cols_to_rename = {k: v for k, v in common_cols_map.items() if k in df_central_cleaned.columns}
        df_central_cleaned.rename(columns=cols_to_rename, inplace=True)

        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col not in df_central_cleaned.columns:
                df_central_cleaned[col] = None
            if col in ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']:
                df_central_cleaned[col] = pd.to_datetime(df_central_cleaned[col], errors='coerce')
            elif col in ['Barcode', 'Vendor number', 'Company code']:
                df_central_cleaned[col] = df_central_cleaned[col].astype(str).replace('nan', '')
            elif df_central_cleaned[col].dtype == 'object':
                df_central_cleaned[col] = df_central_cleaned[col].fillna('')


    except Exception as e:
        return False, f"Error processing central file (Step 2): {e}"
    print(f"--- Central File Status Processing (Step 2) Complete ---")
    return True, df_central_cleaned


def process_central_file_step3_final_merge_and_needs_review(consolidated_main_df, consolidated_workon_df, updated_existing_central_df, final_central_output_file_path, df_pisa_original, df_esm_original, df_pm7_original, region_mapping_df):
    """
    Step 3: Handles barcodes present only in consolidated_main (adds them as new)
            and barcodes present only in central (marks them as 'Needs Review' if not 'Completed').
            Then directly appends consolidated_workon_df.
            Also performs region mapping and final column reordering.
    """
    print(f"\n--- Starting Central File Status Processing (Step 3: Final Merge & Needs Review) ---")

    df_pisa_lookup = clean_column_names(df_pisa_original.copy())
    df_esm_lookup = clean_column_names(df_esm_original.copy())
    df_pm7_lookup = clean_column_names(df_pm7_original.copy())

    df_pisa_indexed = pd.DataFrame()
    if 'barcode' in df_pisa_lookup.columns:
        df_pisa_lookup['barcode'] = df_pisa_lookup['barcode'].astype(str)
        df_pisa_indexed = df_pisa_lookup.set_index('barcode')
    else:
        print("Warning: 'barcode' column not found in cleaned PISA lookup. Cannot perform PISA lookups.")

    df_esm_indexed = pd.DataFrame()
    if 'barcode' in df_esm_lookup.columns:
        df_esm_lookup['barcode'] = df_esm_lookup['barcode'].astype(str)
        df_esm_indexed = df_esm_lookup.set_index('barcode')
    else:
        print("Warning: 'barcode' column not found in cleaned ESM lookup. Cannot perform ESM lookups.")

    df_pm7_indexed = pd.DataFrame()
    if 'barcode' in df_pm7_lookup.columns:
        df_pm7_lookup['barcode'] = df_pm7_lookup['barcode'].astype(str)
        df_pm7_indexed = df_pm7_lookup.set_index('barcode')
    else:
        print("Warning: 'barcode' column not found in cleaned PM7 lookup. Cannot perform PM7 lookups.")

    # --- Process PISA/ESM/PM7 for new records and Needs Review ---
    if 'Barcode' not in consolidated_main_df.columns:
        return False, "Error: 'Barcode' column not found in the main consolidated file. Cannot proceed with final central file processing (Step 3)."
    if 'Barcode' not in updated_existing_central_df.columns or 'Status' not in updated_existing_central_df.columns:
        return False, "Error: 'Barcode' or 'Status' column not found in the updated central file. Cannot update status (Step 3)."

    consolidated_main_barcodes_set = set(consolidated_main_df['Barcode'].unique())
    central_barcodes_set = set(updated_existing_central_df['Barcode'].unique())

    barcodes_to_add = consolidated_main_barcodes_set - central_barcodes_set
    print(f"Found {len(barcodes_to_add)} new barcodes from PISA/ESM/PM7 to add to central.")

    df_new_records_from_main_consolidated = consolidated_main_df[consolidated_main_df['Barcode'].isin(barcodes_to_add)].copy()

    all_new_central_rows_data = []

    for index, row_consolidated in df_new_records_from_main_consolidated.iterrows():
        barcode = str(row_consolidated['Barcode'])
        channel = row_consolidated['Channel']

        # Initialize with values from consolidated_main_df, then try to enrich from original sources
        vendor_name = row_consolidated.get('Vendor Name')
        vendor_number = row_consolidated.get('Vendor number')
        company_code = row_consolidated.get('Company code')
        received_date = row_consolidated.get('Received Date') # This is a datetime object
        processor = row_consolidated.get('Processor')
        category = row_consolidated.get('Category')
        requester = row_consolidated.get('Requester')
        remarks = row_consolidated.get('Remarks')
        region = row_consolidated.get('Region')
        
        # --- PISA Lookup ---
        if channel == 'PISA' and not df_pisa_indexed.empty and barcode in df_pisa_indexed.index:
            pisa_row = df_pisa_indexed.loc[barcode]
            if 'vendor_name' in pisa_row.index and pd.notna(pisa_row['vendor_name']):
                vendor_name = pisa_row['vendor_name']
            if 'vendor_number' in pisa_row.index and pd.notna(pisa_row['vendor_number']):
                vendor_number = pisa_row['vendor_number']
            if 'company_code' in pisa_row.index and pd.notna(pisa_row['company_code']):
                company_code = pisa_row['company_code']
            if 'received_date' in pisa_row.index and pd.notna(pisa_row['received_date']):
                received_date = pd.to_datetime(pisa_row['received_date'], errors='coerce')

        # --- ESM Lookup ---
        elif channel == 'ESM' and not df_esm_indexed.empty and barcode in df_esm_indexed.index:
            esm_row = df_esm_indexed.loc[barcode]
            if 'company_code' in esm_row.index and pd.notna(esm_row['company_code']):
                company_code = esm_row['company_code']
            if 'subcategory' in esm_row.index and pd.notna(esm_row['subcategory']):
                category = esm_row['subcategory']
            if 'vendor_name' in esm_row.index and pd.notna(esm_row['vendor_name']):
                vendor_name = esm_row['vendor_name']
            if 'vendor_number' in esm_row.index and pd.notna(esm_row['vendor_number']):
                vendor_number = esm_row['vendor_number']
            if 'received_date' in esm_row.index and pd.notna(esm_row['received_date']):
                received_date = pd.to_datetime(esm_row['received_date'], errors='coerce')

        # --- PM7 Lookup ---
        elif channel == 'PM7' and not df_pm7_indexed.empty and barcode in df_pm7_indexed.index:
            pm7_row = df_pm7_indexed.loc[barcode]
            if 'vendor_name' in pm7_row.index and pd.notna(pm7_row['vendor_name']):
                vendor_name = pm7_row['vendor_name']
            if 'vendor_number' in pm7_row.index and pd.notna(pm7_row['vendor_number']):
                vendor_number = pm7_row['vendor_number']
            if 'company_code' in pm7_row.index and pd.notna(pm7_row['company_code']):
                company_code = pm7_row['company_code']
            if 'received_date' in pm7_row.index and pd.notna(pm7_row['received_date']):
                received_date = pd.to_datetime(pm7_row['received_date'], errors='coerce')
        
        new_central_row_data = {
            'Barcode': barcode,
            'Processor': processor if processor is not None else '',
            'Channel': channel,
            'Category': category if category is not None else '',
            'Company code': company_code if company_code is not None else '',
            'Region': region if region is not None else '',
            'Vendor number': vendor_number if vendor_number is not None else '',
            'Vendor Name': vendor_name if vendor_name is not None else '',
            'Status': 'New',
            'Received Date': received_date,
            'Re-Open Date': None,
            'Allocation Date': datetime.now(),
            'Clarification Date': None,
            'Completion Date': None,
            'Requester': requester if requester is not None else '',
            'Remarks': remarks if remarks is not None else '',
            'Aging': None,
            'Today': datetime.now()
        }

        all_new_central_rows_data.append(new_central_row_data)

    if all_new_central_rows_data:
        df_new_central_rows = pd.DataFrame(all_new_central_rows_data)
        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col not in df_new_central_rows.columns:
                df_new_central_rows[col] = None
        df_new_central_rows = df_new_central_rows[CONSOLIDATED_OUTPUT_COLUMNS]
    else:
        df_new_central_rows = pd.DataFrame(columns=CONSOLIDATED_OUTPUT_COLUMNS)

    df_final_central = updated_existing_central_df.copy()

    barcodes_for_needs_review = central_barcodes_set - consolidated_main_barcodes_set
    
    needs_review_barcode_mask = df_final_central['Barcode'].isin(barcodes_for_needs_review)
    is_not_completed_status_mask = ~df_final_central['Status'].astype(str).str.strip().str.lower().eq('completed')
    final_needs_review_condition = needs_review_barcode_mask & is_not_completed_status_mask

    df_final_central.loc[final_needs_review_condition, 'Status'] = 'Needs Review'

    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_final_central.columns:
            df_final_central[col] = None
    df_final_central = df_final_central[CONSOLIDATED_OUTPUT_COLUMNS]

    df_final_central = pd.concat([df_final_central, df_new_central_rows], ignore_index=True)

    # --- Append Workon P71 data directly ---
    if not consolidated_workon_df.empty:
        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col not in consolidated_workon_df.columns:
                consolidated_workon_df[col] = None
        consolidated_workon_df = consolidated_workon_df[CONSOLIDATED_OUTPUT_COLUMNS]
        df_final_central = pd.concat([df_final_central, consolidated_workon_df], ignore_index=True)
    else:
        print("No Workon P71 data to append.")

    # --- Handle blank Company Code for PM7 channel (Applies to all new/appended data) ---
    if 'Channel' in df_final_central.columns and 'Company code' in df_final_central.columns and 'Barcode' in df_final_central.columns:
        pm7_blank_cc_mask = (df_final_central['Channel'] == 'PM7') & \
                            (df_final_central['Company code'].astype(str).replace('nan', '').str.strip() == '')

        df_final_central.loc[pm7_blank_cc_mask, 'Company code'] = \
            df_final_central.loc[pm7_blank_cc_mask, 'Barcode'].astype(str).str[:4]
    else:
        print("Warning: 'Channel', 'Company code', or 'Barcode' columns missing. Skipping PM7 Company Code population logic.")

    # --- Apply Region Mapping (Applies to all new/appended data) ---
    if region_mapping_df is None or region_mapping_df.empty:
        df_final_central['Region'] = df_final_central['Region'].fillna('')
    else:
        region_mapping_df = clean_column_names(region_mapping_df.copy())
        if 'r3_coco' not in region_mapping_df.columns or 'region' not in region_mapping_df.columns:
            print("Error: Region mapping file must contain 'r3_coco' and 'region' columns after cleaning. Skipping region mapping.")
            df_final_central['Region'] = df_final_central['Region'].fillna('')
        else:
            region_map = {}
            for idx, row in region_mapping_df.iterrows():
                coco_key = str(row['r3_coco']).strip().upper()
                if coco_key:
                    region_map[coco_key[:4]] = str(row['region']).strip()

            if 'Company code' in df_final_central.columns:
                region_mask_to_fill = df_final_central['Region'].apply(lambda x: pd.isna(x) or str(x).strip() == '')
                
                df_final_central['Company code'] = df_final_central['Company code'].astype(str).str.strip().str.upper().str[:4]
                
                df_final_central.loc[region_mask_to_fill, 'Region'] = df_final_central.loc[region_mask_to_fill, 'Company code'].map(region_map)
                
                df_final_central['Region'] = df_final_central['Region'].fillna('')
            else:
                print("Warning: 'Company code' column not found in final central DataFrame. Cannot apply region mapping.")
                df_final_central['Region'] = df_final_central['Region'].fillna('')

    # Final formatting of dates to MM/DD/YYYY strings and other fills
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col in ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']:
            df_final_central[col] = format_date_to_mdyyyy(df_final_central[col])
        elif df_final_central[col].dtype == 'object':
            df_final_central[col] = df_final_central[col].fillna('')
        elif col in ['Barcode', 'Vendor number', 'Company code']:
            df_final_central[col] = df_final_central[col].astype(str).replace('nan', '')

    df_final_central = df_final_central[CONSOLIDATED_OUTPUT_COLUMNS]

    try:
        df_final_central.to_excel(final_central_output_file_path, index=False)
    except Exception as e:
        return False, f"Error saving final central file (after Step 3): {e}"
    print(f"--- Central File Status Processing (Step 3) Complete ---")
    return True, "Central file processing (Step 3) successful"


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    temp_dir = tempfile.mkdtemp(dir='/tmp')

    session.pop('consolidated_output_path', None)
    session.pop('central_output_path', None)
    session.pop('temp_dir', None)

    session['temp_dir'] = temp_dir

    REGION_MAPPING_FILE_PATH = os.path.join(BASE_DIR, '..', 'company_code_region_mapping.xlsx')

    try:
        uploaded_files = {}
        file_keys = ['pisa_file', 'esm_file', 'pm7_file', 'workon_file', 'central_file']
        for key in file_keys:
            if key not in request.files:
                flash(f'Missing file: "{key}". All five files are required.', 'error')
                return redirect(url_for('index'))
            file = request.files[key]
            if file.filename == '':
                flash(f'No selected file for "{key}". All five files are required.', 'error')
                return redirect(url_for('index'))

            if file and file.filename.lower().endswith('.xlsx'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                uploaded_files[key] = file_path
                flash(f'File "{filename}" uploaded successfully.', 'info')
            else:
                flash(f'Invalid file type for "{key}". Please upload an .xlsx file.', 'error')
                return redirect(url_for('index'))

        pisa_file_path = uploaded_files['pisa_file']
        esm_file_path = uploaded_files['esm_file']
        pm7_file_path = uploaded_files['pm7_file']
        workon_file_path = uploaded_files['workon_file']
        initial_central_file_input_path = uploaded_files['central_file']

        df_pisa_original = None
        df_esm_original = None
        df_pm7_original = None
        df_workon_original = None
        df_region_mapping = None

        try:
            df_pisa_original = pd.read_excel(pisa_file_path)
            df_esm_original = pd.read_excel(esm_file_path)
            df_pm7_original = pd.read_excel(pm7_file_path)
            df_workon_original = pd.read_excel(workon_file_path)

            if os.path.exists(REGION_MAPPING_FILE_PATH):
                df_region_mapping = pd.read_excel(REGION_MAPPING_FILE_PATH)
            else:
                flash(f"Error: Region mapping file not found at {REGION_MAPPING_FILE_PATH}. Region column will be empty.", 'warning')
                df_region_mapping = pd.DataFrame(columns=['R/3 CoCo', 'Region'])

        except Exception as e:
            flash(f"Error loading one or more input Excel files or the region mapping file: {e}. Please ensure all files are valid .xlsx formats and the mapping file exists.", 'error')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            session.pop('temp_dir', None)
            return redirect(url_for('index'))


        today_str = datetime.now().strftime("%d_%m_%Y_%H%M%S")

        # --- Step 1: Consolidate Data (returns main and workon dataframes separately) ---
        consolidated_main_output_filename = f'ConsolidatedMainData_{today_str}.xlsx'
        consolidated_main_output_file_path = os.path.join(temp_dir, consolidated_main_output_filename)
        success, results = consolidate_data_process(
            df_pisa_original, df_esm_original, df_pm7_original, df_workon_original, consolidated_main_output_file_path
        )

        if not success:
            flash(f'Consolidation Error: {results}', 'error')
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            session.pop('temp_dir', None)
            return redirect(url_for('index'))
        df_consolidated_main, df_consolidated_workon = results
        session['consolidated_output_path'] = consolidated_main_output_file_path
        flash('Data consolidation from all sources completed successfully!', 'success')
        

        # --- Step 2: Update existing central file records based on main consolidation ---
        success, result_df = process_central_file_step2_update_existing(
            df_consolidated_main, initial_central_file_input_path
        )
        if not success:
            flash(f'Central File Processing (Step 2) Error: {result_df}', 'error')
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            session.pop('temp_dir', None)
            return redirect(url_for('index'))
        df_central_updated_existing = result_df

        # --- Step 3: Final Merge (PISA/ESM/PM7 new/needs review, then append Workon, apply global logic) ---
        final_central_output_filename = f'CentralFile_FinalOutput_{today_str}.xlsx'
        final_central_output_file_path = os.path.join(temp_dir, final_central_output_filename)
        success, message = process_central_file_step3_final_merge_and_needs_review(
            df_consolidated_main, df_consolidated_workon, df_central_updated_existing, final_central_output_file_path,
            df_pisa_original, df_esm_original, df_pm7_original, df_region_mapping
        )
        if not success:
            flash(f'Central File Processing (Step 3) Error: {message}', 'error')
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            session.pop('temp_dir', None)
            return redirect(url_for('index'))
        flash('Central file finalized successfully!', 'success')
        session['central_output_path'] = final_central_output_file_path

        return render_template('index.html',
                                central_download_link=url_for('download_file', filename=os.path.basename(final_central_output_file_path))
                              )

    except Exception as e:
        flash(f'An unhandled error occurred during processing: {e}', 'error')
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        session.pop('temp_dir', None)
        return redirect(url_for('index'))
    finally:
        pass


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path_in_temp = None
    temp_dir = session.get('temp_dir')

    if not temp_dir:
        flash('File not found for download or session expired. Please re-run the process.', 'error')
        return redirect(url_for('index'))

    consolidated_session_path = session.get('consolidated_output_path')
    central_session_path = session.get('central_output_path')

    if consolidated_session_path and os.path.basename(consolidated_session_path) == filename:
        file_path_in_temp = os.path.join(temp_dir, filename)
    elif central_session_path and os.path.basename(central_session_path) == filename:
        file_path_in_temp = os.path.join(temp_dir, filename)
    else:
        pass

    if file_path_in_temp and os.path.exists(file_path_in_temp):
        try:
            response = send_file(
                file_path_in_temp,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=filename
            )
            return response
        except Exception as e:
            flash(f'Error providing download: {e}. Please try again.', 'error')
            return redirect(url_for('index'))
    else:
        flash('File not found for download or session expired. Please re-run the process.', 'error')
        return redirect(url_for('index'))

@app.route('/cleanup_session', methods=['GET'])
def cleanup_session():
    temp_dir = session.get('temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            flash('Temporary files cleaned up.', 'info')
        except OSError as e:
            flash(f'Error cleaning up temporary files: {e}', 'error')
    session.pop('temp_dir', None)
    session.pop('consolidated_output_path', None)
    session.pop('central_output_path', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
