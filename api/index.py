import os
import pandas as pd
from datetime import datetime
import warnings
import shutil
import tempfile
import re
import io  # <-- ADD THIS IMPORT

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

# --- Helper Functions (No changes needed here) ---

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
    1\. Lowercasing all characters.
    2\. Replacing spaces with underscores.
    3\. Removing special characters (keeping only alphanumeric and underscores).
    4\. Removing leading/trailing underscores.
    """
    new_columns = []
    for col in df.columns:
        col = str(col).strip().lower()
        col = re.sub(r'\\s+', '_', col)
        col = re.sub(r'[^a-z0-9_]', '', col)
        col = col.strip('_')
        new_columns.append(col)
    df.columns = new_columns
    return df

def calculate_aging(df):
    """
    Calculates the 'Aging' for each row based on 'Received Date' and 'Today'.
    Handles various date formats and NaN values gracefully.
    'Today' is expected to be a datetime object.
    """
    if 'Received Date' in df.columns and 'Today' in df.columns:
        df['Received_Date_dt'] = pd.to_datetime(df['Received Date'], errors='coerce')
        df['Today_dt'] = pd.to_datetime(df['Today'], errors='coerce')
        valid_dates_mask = df['Received_Date_dt'].notna() & df['Today_dt'].notna()
        df.loc[valid_dates_mask, 'Aging'] = (df.loc[valid_dates_mask, 'Today_dt'] - df.loc[valid_dates_mask, 'Received_Date_dt']).dt.days
        df['Aging'] = df['Aging'].fillna('').astype(str)
        df = df.drop(columns=['Received_Date_dt', 'Today_dt'])
    else:
        df['Aging'] = ''
    return df

def consolidate_data_process(df_pisa, df_esm, df_pm7, df_smd, consolidated_output_file_path):
    """
    Reads PISA, ESM, PM7, and SMD Excel files (now passed as DFs), filters PISA, consolidates data,
    and saves it to a new Excel file.
    """
    print("Starting data consolidation process...")
    print("All input DataFrames loaded successfully!")
    df_pisa = clean_column_names(df_pisa.copy())
    df_esm = clean_column_names(df_esm.copy())
    df_pm7 = clean_column_names(df_pm7.copy())
    df_smd = clean_column_names(df_smd.copy())

    allowed_pisa_users = ["Goswami Sonali", "Patil Jayapal Gowd", "Ranganath Chilamakuri","Sridhar Divya","Sunitha S","Varunkumar N"]
    if 'assigned_user' in df_pisa.columns:
        original_pisa_count = len(df_pisa)
        df_pisa_filtered = df_pisa[df_pisa['assigned_user'].isin(allowed_pisa_users)].copy()
        print(f"\\nPISA file filtered. Original records: {original_pisa_count}, Records after filter: {len(df_pisa_filtered)}")
    else:
        print("\\nWarning: 'assigned_user' column not found in PISA file (after cleaning). No filter applied.")
        df_pisa_filtered = df_pisa.copy()

    all_consolidated_rows = []
    today_date = datetime.now()

    # --- PISA Processing ---
    if 'barcode' not in df_pisa_filtered.columns:
        print("Error: 'barcode' column not found in PISA file (after cleaning). Skipping PISA processing.")
    else:
        df_pisa_filtered['barcode'] = df_pisa_filtered['barcode'].astype(str)
        for index, row in df_pisa_filtered.iterrows():
            new_row = {
                'Barcode': row['barcode'], 'Company code': row.get('company_code'),
                'Vendor number': row.get('vendor_number'), 'Received Date': row.get('received_date'),
                'Completion Date': None, 'Status': None , 'Today': today_date, 'Channel': 'PISA',
                'Vendor Name': row.get('vendor_name'), 'Re-Open Date': None, 'Allocation Date': None,
                'Requester': None, 'Clarification Date': None, 'Aging': None, 'Remarks': None,
                'Region': None, 'Processor': None, 'Category': None
            }
            all_consolidated_rows.append(new_row)
        print(f"Collected {len(df_pisa_filtered)} rows from PISA.")

    # --- ESM Processing ---
    if 'barcode' not in df_esm.columns:
        print("Error: 'barcode' column not found in ESM file (after cleaning). Skipping ESM processing.")
    else:
        df_esm['barcode'] = df_esm['barcode'].astype(str)
        for index, row in df_esm.iterrows():
            new_row = {
                'Barcode': row['barcode'], 'Received Date': row.get('received_date'),
                'Status': row.get('state'), 'Requester': row.get('opened_by'),
                'Completion Date': row.get('closed') if pd.notna(row.get('closed')) else None,
                'Re-Open Date': row.get('updated') if (row.get('state') or '').lower() == 'reopened' else None,
                'Today': today_date, 'Remarks': row.get('short_description'), 'Channel': 'ESM',
                'Company code': None,'Vendor Name': None, 'Vendor number': None, 'Allocation Date': None,
                'Clarification Date': None, 'Aging': None, 'Region': None, 'Processor': None, 'Category': None
            }
            all_consolidated_rows.append(new_row)
        print(f"Collected {len(df_esm)} rows from ESM.")

    # --- PM7 Processing ---
    if 'barcode' not in df_pm7.columns:
        print("Error: 'barcode' column not found in PM7 file (after cleaning). Skipping PM7 processing.")
    else:
        df_pm7['barcode'] = df_pm7['barcode'].astype(str)
        for index, row in df_pm7.iterrows():
            new_row = {
                'Barcode': row['barcode'], 'Vendor Name': row.get('vendor_name'),
                'Vendor number': row.get('vendor_number'), 'Received Date': row.get('received_date'),
                'Status': row.get('task'), 'Today': today_date, 'Channel': 'PM7',
                'Company code': row.get('company_code'), 'Re-Open Date': None, 'Allocation Date': None,
                'Completion Date': None, 'Requester': None, 'Clarification Date': None, 'Aging': None,
                'Remarks': None, 'Region': None, 'Processor': None, 'Category': None
            }
            all_consolidated_rows.append(new_row)
        print(f"Collected {len(df_pm7)} rows from PM7.")

    # --- SMD Processing ---
    if 'barcode' not in df_smd.columns:
        print("Error: 'barcode' column not found in SMD file (after cleaning). Skipping SMD processing.")
    else:
        df_smd['barcode'] = df_smd['barcode'].astype(str)
        for index, row in df_smd.iterrows():
            new_row = {
                'Barcode': row['barcode'], 'Company code': row.get('ekorg'),
                'Region': row.get('material_field'), 'Vendor number': row.get('pmd_sno'),
                'Vendor Name': row.get('supplier_name'), 'Received Date': row.get('request_date'),
                'Requester': row.get('requested_by'), 'Today': today_date, 'Channel': 'SMD',
                'Status': None, 'Completion Date': None, 'Re-Open Date': None, 'Allocation Date': None,
                'Clarification Date': None, 'Aging': None, 'Remarks': None, 'Processor': None, 'Category': None
            }
            all_consolidated_rows.append(new_row)
        print(f"Collected {len(df_smd)} rows from SMD.")

    if not all_consolidated_rows:
        return False, "No data collected for consolidation."

    df_consolidated = pd.DataFrame(all_consolidated_rows)
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_consolidated.columns:
            df_consolidated[col] = None
    df_consolidated = df_consolidated[CONSOLIDATED_OUTPUT_COLUMNS]
    df_consolidated = calculate_aging(df_consolidated)

    date_cols_to_process = ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']
    for col in df_consolidated.columns:
        if col in date_cols_to_process:
            df_consolidated[col] = format_date_to_mdyyyy(df_consolidated[col])
        else:
            if df_consolidated[col].dtype == 'object':
                df_consolidated[col] = df_consolidated[col].fillna('')
            elif col in ['Barcode', 'Company code', 'Vendor number', 'Aging']:
                df_consolidated[col] = df_consolidated[col].astype(str).replace('nan', '')

    try:
        df_consolidated.to_excel(consolidated_output_file_path, index=False)
        print(f"Consolidated file saved to: {consolidated_output_file_path}")
    except Exception as e:
        return False, f"Error saving consolidated file: {e}"

    print("--- Consolidated Data Process Complete ---")
    return True, df_consolidated

def process_central_file_step2_update_existing(consolidated_df, central_file_input_path):
    """
    Step 2: Updates status of *existing* central file records based on consolidated data.
    """
    print(f"\\n--- Starting Central File Status Processing (Step 2: Update Existing Barcodes) ---")
    try:
        converters = {'Barcode': str, 'Vendor number': str, 'Company code': str}
        df_central = pd.read_excel(central_file_input_path, converters=converters, keep_default_na=False)
        df_central_cleaned = clean_column_names(df_central.copy())
        print("Consolidated (DF) and Central (file) loaded successfully for Step 2!")
    except Exception as e:
        return False, f"Error loading Consolidated (DF) or Central (file) for processing (Step 2): {e}"

    if 'Barcode' not in consolidated_df.columns:
        return False, "Error: 'Barcode' column not found in the consolidated file. Cannot proceed with central file processing (Step 2)."
    if 'barcode' not in df_central_cleaned.columns or 'status' not in df_central_cleaned.columns:
        return False, "Error: 'barcode' or 'status' column not found in the central file after cleaning. Cannot update status (Step 2)."

    consolidated_df['Barcode'] = consolidated_df['Barcode'].astype(str)
    df_central_cleaned['barcode'] = df_central_cleaned['barcode'].astype(str)
    df_central_cleaned['Barcode_compare'] = df_central_cleaned['barcode']
    consolidated_barcodes_set = set(consolidated_df['Barcode'].unique())
    print(f"Found {len(consolidated_barcodes_set)} unique barcodes in the consolidated file for Step 2.")

    def transform_status_if_barcode_exists(row):
        central_barcode = str(row['Barcode_compare'])
        original_central_status = row['status']
        if central_barcode in consolidated_barcodes_set:
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

        date_cols_in_central_file = [
            'Received Date', 'Re-Open Date', 'Allocation Date',
            'Completion Date', 'Clarification Date', 'Today'
        ]
        for col in df_central_cleaned.columns:
            if col in date_cols_in_central_file:
                df_central_cleaned[col] = format_date_to_mdyyyy(df_central_cleaned[col])
            elif df_central_cleaned[col].dtype == 'object':
                df_central_cleaned[col] = df_central_cleaned[col].fillna('')
            elif col in ['Barcode', 'Vendor number', 'Aging']:
                df_central_cleaned[col] = df_central_cleaned[col].astype(str).replace('nan', '')
            if col == 'Company code':
                 df_central_cleaned[col] = df_central_cleaned[col].astype(str).replace('nan', '')
        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col not in df_central_cleaned.columns:
                df_central_cleaned[col] = None
    except Exception as e:
        return False, f"Error processing central file (Step 2): {e}"

    print(f"--- Central File Status Processing (Step 2) Complete ---")
    return True, df_central_cleaned

def process_central_file_step3_final_merge_and_needs_review(consolidated_df, updated_existing_central_df, final_central_output_file_path, df_pisa_original, df_esm_original, df_pm7_original, df_smd_original, region_mapping_df):
    """
    Step 3: Handles barcodes present only in consolidated (adds them as new)
            and barcodes present only in central (marks them as 'Needs Review' if not 'Completed').
            Also performs region mapping and final column reordering.
    """
    print(f"\\n--- Starting Central File Status Processing (Step 3: Final Merge & Needs Review) ---")
    # ... (rest of the function is the same until the very end)
    df_pisa_lookup = clean_column_names(df_pisa_original.copy())
    df_esm_lookup = clean_column_names(df_esm_original.copy())
    df_pm7_lookup = clean_column_names(df_pm7_original.copy())
    df_smd_lookup = clean_column_names(df_smd_original.copy()) # Added SMD lookup

    df_pisa_indexed = pd.DataFrame()
    if 'barcode' in df_pisa_lookup.columns:
        df_pisa_lookup['barcode'] = df_pisa_lookup['barcode'].astype(str)
        df_pisa_indexed = df_pisa_lookup.set_index('barcode')
        print(f"PISA lookup indexed by 'barcode'.")
    else:
        print("Warning: 'barcode' column not found in cleaned PISA lookup. Cannot perform PISA lookups.")

    df_esm_indexed = pd.DataFrame()
    if 'barcode' in df_esm_lookup.columns:
        df_esm_lookup['barcode'] = df_esm_lookup['barcode'].astype(str)
        df_esm_indexed = df_esm_lookup.set_index('barcode')
        print(f"ESM lookup indexed by 'barcode'.")
    else:
        print("Warning: 'barcode' column not found in cleaned ESM lookup. Cannot perform ESM lookups.")

    df_pm7_indexed = pd.DataFrame()
    if 'barcode' in df_pm7_lookup.columns:
        df_pm7_lookup['barcode'] = df_pm7_lookup['barcode'].astype(str)
        df_pm7_indexed = df_pm7_lookup.set_index('barcode')
        print(f"PM7 lookup indexed by 'barcode'.")
    else:
        print("Warning: 'barcode' column not found in cleaned PM7 lookup. Cannot perform PM7 lookups.")

    df_smd_indexed = pd.DataFrame() # Added SMD lookup
    if 'barcode' in df_smd_lookup.columns:
        df_smd_lookup['barcode'] = df_smd_lookup['barcode'].astype(str)
        df_smd_indexed = df_smd_lookup.set_index('barcode')
        print(f"SMD lookup indexed by 'barcode'.")
    else:
        print("Warning: 'barcode' column not found in cleaned SMD lookup. Cannot perform SMD lookups.")

    if 'Barcode' not in consolidated_df.columns:
        return False, "Error: 'Barcode' column not found in the consolidated file. Cannot proceed with final central file processing (Step 3)."
    if 'Barcode' not in updated_existing_central_df.columns or 'Status' not in updated_existing_central_df.columns:
        return False, "Error: 'Barcode' or 'Status' column not found in the updated central file. Cannot update status (Step 3)."

    consolidated_barcodes_set = set(consolidated_df['Barcode'].unique())
    central_barcodes_set = set(updated_existing_central_df['Barcode'].unique())
    barcodes_to_add = consolidated_barcodes_set - central_barcodes_set
    print(f"Found {len(barcodes_to_add)} new barcodes in consolidated file to add to central.")
    df_new_records_from_consolidated = consolidated_df[consolidated_df['Barcode'].isin(barcodes_to_add)].copy()
    all_new_central_rows_data = []

    for index, row_consolidated in df_new_records_from_consolidated.iterrows():
        barcode = row_consolidated['Barcode']
        channel = row_consolidated['Channel']
        vendor_name = row_consolidated.get('Vendor Name')
        vendor_number = row_consolidated.get('Vendor number')
        company_code = row_consolidated.get('Company code')
        received_date = row_consolidated.get('Received Date')
        processor = row_consolidated.get('Processor')
        category = row_consolidated.get('Category')
        region = row_consolidated.get('Region')
        requester = row_consolidated.get('Requester')
        
        # --- PISA Lookup ---
        if channel == 'PISA' and not df_pisa_indexed.empty and barcode in df_pisa_indexed.index:
            pisa_row = df_pisa_indexed.loc[barcode]
            if 'vendor_name' in pisa_row.index and pd.notna(pisa_row['vendor_name']): vendor_name = pisa_row['vendor_name']
            if 'vendor_number' in pisa_row.index and pd.notna(pisa_row['vendor_number']): vendor_number = pisa_row['vendor_number']
            if 'company_code' in pisa_row.index and pd.notna(pisa_row['company_code']): company_code = pisa_row['company_code']
            if 'received_date' in pisa_row.index and pd.notna(pisa_row['received_date']): received_date = pisa_row['received_date']

        # --- ESM Lookup ---
        elif channel == 'ESM' and not df_esm_indexed.empty and barcode in df_esm_indexed.index:
            esm_row = df_esm_indexed.loc[barcode]
            if 'company_code' in esm_row.index and pd.notna(esm_row['company_code']): company_code = esm_row['company_code']
            if 'subcategory' in esm_row.index and pd.notna(esm_row['subcategory']): category = esm_row['subcategory']
            if 'vendor_name' in esm_row.index and pd.notna(esm_row['vendor_name']): vendor_name = esm_row['vendor_name']
            if 'vendor_number' in esm_row.index and pd.notna(esm_row['vendor_number']): vendor_number = esm_row['vendor_number']
            if 'received_date' in esm_row.index and pd.notna(esm_row['received_date']): received_date = esm_row['received_date']

        # --- PM7 Lookup ---
        elif channel == 'PM7' and not df_pm7_indexed.empty and barcode in df_pm7_indexed.index:
            pm7_row = df_pm7_indexed.loc[barcode]
            if 'vendor_name' in pm7_row.index and pd.notna(pm7_row['vendor_name']): vendor_name = pm7_row['vendor_name']
            if 'vendor_number' in pm7_row.index and pd.notna(pm7_row['vendor_number']): vendor_number = pm7_row['vendor_number']
            if 'company_code' in pm7_row.index and pd.notna(pm7_row['company_code']): company_code = pm7_row['company_code']
            if 'received_date' in pm7_row.index and pd.notna(pm7_row['received_date']): received_date = pm7_row['received_date']

        # --- SMD Lookup ---
        elif channel == 'SMD' and not df_smd_indexed.empty and barcode in df_smd_indexed.index:
            smd_row = df_smd_indexed.loc[barcode]
            if 'ekorg' in smd_row.index and pd.notna(smd_row['ekorg']): company_code = smd_row['ekorg']
            if 'material_field' in smd_row.index and pd.notna(smd_row['material_field']): region = smd_row['material_field']
            if 'pmd_sno' in smd_row.index and pd.notna(smd_row['pmd_sno']): vendor_number = smd_row['pmd_sno']
            if 'supplier_name' in smd_row.index and pd.notna(smd_row['supplier_name']): vendor_name = smd_row['supplier_name']
            if 'request_date' in smd_row.index and pd.notna(smd_row['request_date']): received_date = smd_row['request_date']
            if 'requested_by' in smd_row.index and pd.notna(smd_row['requested_by']): requester = smd_row['requested_by']

        new_central_row_data = row_consolidated.to_dict()
        new_central_row_data['Vendor Name'] = vendor_name if vendor_name is not None else ''
        new_central_row_data['Vendor number'] = vendor_number if vendor_number is not None else ''
        new_central_row_data['Company code'] = company_code if company_code is not None else ''
        new_central_row_data['Received Date'] = received_date
        new_central_row_data['Status'] = 'New'
        new_central_row_data['Allocation Date'] = datetime.now().strftime("%m/%d/%Y")
        new_central_row_data['Processor'] = processor if processor is not None else ''
        new_central_row_data['Category'] = category if category is not None else ''
        new_central_row_data['Region'] = region if region is not None else ''
        new_central_row_data['Requester'] = requester if requester is not None else ''
        all_new_central_rows_data.append(new_central_row_data)

    if all_new_central_rows_data:
        df_new_central_rows = pd.DataFrame(all_new_central_rows_data)
        for col in CONSOLIDATED_OUTPUT_COLUMNS:
            if col not in df_new_central_rows.columns: df_new_central_rows[col] = None
        df_new_central_rows = df_new_central_rows[CONSOLIDATED_OUTPUT_COLUMNS]
    else:
        df_new_central_rows = pd.DataFrame(columns=CONSOLIDATED_OUTPUT_COLUMNS)

    for col in df_new_central_rows.columns:
        if df_new_central_rows[col].dtype == 'object':
            df_new_central_rows[col] = df_new_central_rows[col].fillna('')
        elif col in ['Barcode', 'Company code', 'Vendor number', 'Aging']:
            df_new_central_rows[col] = df_new_central_rows[col].astype(str).replace('nan', '')

    barcodes_for_needs_review = central_barcodes_set - consolidated_barcodes_set
    print(f"Found {len(barcodes_for_needs_review)} barcodes in central not in consolidated.")
    df_final_central = updated_existing_central_df.copy()
    needs_review_barcode_mask = df_final_central['Barcode'].isin(barcodes_for_needs_review)
    is_not_completed_status_mask = ~df_final_central['Status'].astype(str).str.strip().str.lower().eq('completed')
    final_needs_review_condition = needs_review_barcode_mask & is_not_completed_status_mask
    df_final_central.loc[final_needs_review_condition, 'Status'] = 'Needs Review'
    print(f"Updated {final_needs_review_condition.sum()} records to 'Needs Review' where status was not 'Completed'.")

    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_final_central.columns: df_final_central[col] = None
    df_final_central = df_final_central[CONSOLIDATED_OUTPUT_COLUMNS]
    df_final_central = pd.concat([df_final_central, df_new_central_rows], ignore_index=True)

    # --- NEW LOGIC START: Handle blank Company Code for PM7 channel ---
    print("\\n--- Applying PM7 Company Code population logic ---")
    if 'Channel' in df_final_central.columns and 'Company code' in df_final_central.columns and 'Barcode' in df_final_central.columns:
        pm7_blank_cc_mask = (df_final_central['Channel'] == 'PM7') & \
                            (df_final_central['Company code'].astype(str).replace('nan', '').str.strip() == '')
        df_final_central.loc[pm7_blank_cc_mask, 'Company code'] = \
            df_final_central.loc[pm7_blank_cc_mask, 'Barcode'].astype(str).str[:4]
        print(f"Populated Company Code for {pm7_blank_cc_mask.sum()} PM7 records based on Barcode.")
    else:
        print("Warning: 'Channel', 'Company code', or 'Barcode' columns missing. Skipping PM7 Company Code population logic.")

    # --- NEW REGION MAPPING LOGIC ---
    print("\\n--- Applying Region Mapping ---")
    if region_mapping_df is None or region_mapping_df.empty:
        print("Warning: Region mapping file not provided or is empty. Region column will not be populated by external mapping.")
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
            print(f"Loaded {len(region_map)} unique R/3 CoCo -> Region mappings.")
            if 'Company code' in df_final_central.columns:
                empty_region_mask = df_final_central['Region'].astype(str).str.strip() == ''
                df_final_central.loc[empty_region_mask, 'Company code_temp'] = \
                    df_final_central.loc[empty_region_mask, 'Company code'].astype(str).str.strip().str.upper().str[:4]
                df_final_central.loc[empty_region_mask, 'Region'] = \
                    df_final_central.loc[empty_region_mask, 'Company code_temp'].map(region_map).fillna(df_final_central.loc[empty_region_mask, 'Region'])
                df_final_central = df_final_central.drop(columns=['Company code_temp'])
                df_final_central['Region'] = df_final_central['Region'].fillna('')
                print("Region mapping applied successfully to empty 'Region' cells.")
            else:
                print("Warning: 'Company code' column not found in final central DataFrame. Cannot apply region mapping.")
                df_final_central['Region'] = df_final_central['Region'].fillna('')

    date_cols_in_central_file = [
        'Received Date', 'Re-Open Date', 'Allocation Date',
        'Completion Date', 'Clarification Date', 'Today'
    ]
    for col in df_final_central.columns:
        if col in date_cols_in_central_file:
            df_final_central[col] = format_date_to_mdyyyy(df_final_central[col])
        elif df_final_central[col].dtype == 'object':
            df_final_central[col] = df_final_central[col].fillna('')
        elif col in ['Barcode', 'Vendor number', 'Aging']:
            df_final_central[col] = df_final_central[col].astype(str).replace('nan', '')

    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_final_central.columns:
            df_final_central[col] = ''
    df_final_central = df_final_central[CONSOLIDATED_OUTPUT_COLUMNS]

    # --- MODIFICATION START ---
    # If a path is provided, save the file. This maintains original behavior if needed locally.
    if final_central_output_file_path:
        try:
            df_final_central.to_excel(final_central_output_file_path, index=False)
            print(f"Final central file (after Step 3) saved to: {final_central_output_file_path}")
            print(f"Total rows in final central file (after Step 3): {len(df_final_central)}")
        except Exception as e:
            return False, f"Error saving final central file (after Step 3): {e}"

    print(f"--- Central File Status Processing (Step 3) Complete ---")
    # Always return the dataframe for in-memory processing
    return True, df_final_central
    # --- MODIFICATION END ---


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    # --- MODIFICATION START: Entire function is streamlined ---
    temp_dir = tempfile.mkdtemp(dir='/tmp')
    
    # Region mapping file is in the project root
    REGION_MAPPING_FILE_PATH = os.path.join(BASE_DIR, '..', 'company_code_region_mapping.xlsx')

    try:
        uploaded_files = {}
        # NOTE: Your HTML uses 'smd_data_file' but your code expects 'smd_file'. I've used 'smd_file' here.
        # You should make them consistent. I am also treating 'smd_file' as optional.
        required_keys = ['pisa_file', 'esm_file', 'pm7_file', 'central_file']
        optional_keys = ['smd_file']

        for key in required_keys:
            if key not in request.files or request.files[key].filename == '':
                flash(f'Missing required file: "{key}".', 'error')
                return redirect(url_for('index'))
            file = request.files[key]
            if file and file.filename.lower().endswith('.xlsx'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                uploaded_files[key] = file_path
            else:
                flash(f'Invalid file type for "{key}". Please upload an .xlsx file.', 'error')
                return redirect(url_for('index'))

        # Handle optional SMD file
        if 'smd_file' in request.files and request.files['smd_file'].filename != '':
            file = request.files['smd_file']
            if file.filename.lower().endswith('.xlsx'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                uploaded_files['smd_file'] = file_path
            else:
                 flash('Invalid file type for "smd_file". Please upload an .xlsx file.', 'error')
                 return redirect(url_for('index'))
        else:
            # Create an empty placeholder if SMD is not provided
            uploaded_files['smd_file'] = os.path.join(temp_dir, 'empty_smd.xlsx')
            pd.DataFrame().to_excel(uploaded_files['smd_file'])


        pisa_file_path = uploaded_files['pisa_file']
        esm_file_path = uploaded_files['esm_file']
        pm7_file_path = uploaded_files['pm7_file']
        smd_file_path = uploaded_files['smd_file']
        initial_central_file_input_path = uploaded_files['central_file']

        df_pisa_original, df_esm_original, df_pm7_original, df_smd_original, df_region_mapping = (None,)*5

        try:
            df_pisa_original = pd.read_excel(pisa_file_path)
            df_esm_original = pd.read_excel(esm_file_path)
            df_pm7_original = pd.read_excel(pm7_file_path)
            df_smd_original = pd.read_excel(smd_file_path)
            if os.path.exists(REGION_MAPPING_FILE_PATH):
                df_region_mapping = pd.read_excel(REGION_MAPPING_FILE_PATH)
                print(f"Successfully loaded region mapping file from: {REGION_MAPPING_FILE_PATH}")
            else:
                flash(f"Warning: Region mapping file not found at {REGION_MAPPING_FILE_PATH}.", 'warning')
                df_region_mapping = pd.DataFrame() # Empty dataframe
        except Exception as e:
            flash(f"Error loading one or more input Excel files: {e}", 'error')
            return redirect(url_for('index'))

        today_str = datetime.now().strftime("%d_%m_%Y_%H%M%S")

        # --- Step 1: Consolidate Data ---
        consolidated_output_filename = f'ConsolidatedData_{today_str}.xlsx'
        consolidated_output_file_path = os.path.join(temp_dir, consolidated_output_filename)
        success, result = consolidate_data_process(df_pisa_original, df_esm_original, df_pm7_original, df_smd_original, consolidated_output_file_path)
        if not success:
            flash(f'Consolidation Error: {result}', 'error')
            return redirect(url_for('index'))
        df_consolidated = result

        # --- Step 2: Update existing central file records ---
        success, result_df = process_central_file_step2_update_existing(df_consolidated, initial_central_file_input_path)
        if not success:
            flash(f'Central File Processing (Step 2) Error: {result_df}', 'error')
            return redirect(url_for('index'))
        df_central_updated_existing = result_df

        # --- Step 3: Final Merge (returns a DataFrame) ---
        success, final_df = process_central_file_step3_final_merge_and_needs_review(
            df_consolidated, df_central_updated_existing, None,  # Pass None for file path
            df_pisa_original, df_esm_original, df_pm7_original, df_smd_original, df_region_mapping
        )
        if not success:
            flash(f'Central File Processing (Step 3) Error: {final_df}', 'error')
            return redirect(url_for('index'))
        
        # --- Create Excel file in-memory and send it ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False)
        output.seek(0)
        
        final_central_output_filename = f'CentralFile_FinalOutput_{today_str}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=final_central_output_filename
        )

    except Exception as e:
        flash(f'An unhandled error occurred during processing: {e}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))
    finally:
        # Clean up the temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        session.clear()
    # --- MODIFICATION END ---


# --- REMOVED OBSOLETE ROUTES ---
# The /download and /cleanup_session routes are no longer needed with this new stateless approach.

if __name__ == '__main__':
    app.run(debug=True)

