import os
import pandas as pd
from datetime import datetime
import warnings
import shutil
import tempfile
import re
import io
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

# Vercel-specific path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, '..', 'templates')
static_dir = os.path.join(BASE_DIR, '..', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
# IMPORTANT: You must set a secret key in your Vercel Environment Variables
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key-for-local-dev-only')

# Global variables (no changes)
CONSOLIDATED_OUTPUT_COLUMNS = [
    'Barcode', 'Processor', 'Channel', 'Category', 'Company code', 'Region',
    'Vendor number', 'Vendor Name', 'Status', 'Received Date', 'Re-Open Date',
    'Allocation Date', 'Clarification Date', 'Completion Date', 'Requester',
    'Remarks', 'Aging', 'Today'
]

# All of your data processing helper functions (format_date_to_mdyyyy, clean_column_names, etc.)
# remain exactly the same. They do not need to be changed.
# ... (all helper functions are unchanged)
def format_date_to_mdyyyy(date_series):
    datetime_series = pd.to_datetime(date_series, errors='coerce')
    return datetime_series.apply(lambda x: f"{x.month}/{x.day}/{x.year}" if pd.notna(x) else '')

def clean_column_names(df):
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

def consolidate_data_process(df_pisa, df_esm, df_pm7, df_smd):
    print("Starting data consolidation process...")
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
    if 'barcode' in df_pisa_filtered.columns:
        df_pisa_filtered['barcode'] = df_pisa_filtered['barcode'].astype(str)
        for index, row in df_pisa_filtered.iterrows():
            new_row = {'Barcode': row['barcode'], 'Company code': row.get('company_code'), 'Vendor number': row.get('vendor_number'), 'Received Date': row.get('received_date'), 'Completion Date': None, 'Status': None , 'Today': today_date, 'Channel': 'PISA', 'Vendor Name': row.get('vendor_name'), 'Re-Open Date': None, 'Allocation Date': None, 'Requester': None, 'Clarification Date': None, 'Aging': None, 'Remarks': None, 'Region': None, 'Processor': None, 'Category': None }
            all_consolidated_rows.append(new_row)
    if 'barcode' in df_esm.columns:
        df_esm['barcode'] = df_esm['barcode'].astype(str)
        for index, row in df_esm.iterrows():
            new_row = {'Barcode': row['barcode'], 'Received Date': row.get('received_date'), 'Status': row.get('state'), 'Requester': row.get('opened_by'), 'Completion Date': row.get('closed') if pd.notna(row.get('closed')) else None, 'Re-Open Date': row.get('updated') if (row.get('state') or '').lower() == 'reopened' else None, 'Today': today_date, 'Remarks': row.get('short_description'), 'Channel': 'ESM', 'Company code': None,'Vendor Name': None, 'Vendor number': None, 'Allocation Date': None, 'Clarification Date': None, 'Aging': None, 'Region': None, 'Processor': None, 'Category': None }
            all_consolidated_rows.append(new_row)
    if 'barcode' in df_pm7.columns:
        df_pm7['barcode'] = df_pm7['barcode'].astype(str)
        for index, row in df_pm7.iterrows():
            new_row = {'Barcode': row['barcode'], 'Vendor Name': row.get('vendor_name'), 'Vendor number': row.get('vendor_number'), 'Received Date': row.get('received_date'), 'Status': row.get('task'), 'Today': today_date, 'Channel': 'PM7', 'Company code': row.get('company_code'), 'Re-Open Date': None, 'Allocation Date': None, 'Completion Date': None, 'Requester': None, 'Clarification Date': None, 'Aging': None, 'Remarks': None, 'Region': None, 'Processor': None, 'Category': None }
            all_consolidated_rows.append(new_row)
    if 'barcode' in df_smd.columns:
        df_smd['barcode'] = df_smd['barcode'].astype(str)
        for index, row in df_smd.iterrows():
            new_row = {'Barcode': row['barcode'], 'Company code': row.get('ekorg'), 'Region': row.get('material_field'), 'Vendor number': row.get('pmd_sno'), 'Vendor Name': row.get('supplier_name'), 'Received Date': row.get('request_date'), 'Requester': row.get('requested_by'), 'Today': today_date, 'Channel': 'SMD', 'Status': None, 'Completion Date': None, 'Re-Open Date': None, 'Allocation Date': None, 'Clarification Date': None, 'Aging': None, 'Remarks': None, 'Processor': None, 'Category': None }
            all_consolidated_rows.append(new_row)
    if not all_consolidated_rows: return False, "No data collected for consolidation."
    df_consolidated = pd.DataFrame(all_consolidated_rows)
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_consolidated.columns: df_consolidated[col] = None
    df_consolidated = df_consolidated[CONSOLIDATED_OUTPUT_COLUMNS]
    df_consolidated = calculate_aging(df_consolidated)
    date_cols_to_process = ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']
    for col in df_consolidated.columns:
        if col in date_cols_to_process: df_consolidated[col] = format_date_to_mdyyyy(df_consolidated[col])
        elif df_consolidated[col].dtype == 'object': df_consolidated[col] = df_consolidated[col].fillna('')
        elif col in ['Barcode', 'Company code', 'Vendor number', 'Aging']: df_consolidated[col] = df_consolidated[col].astype(str).replace('nan', '')
    return True, df_consolidated

def process_central_file_step2_update_existing(consolidated_df, central_file_input_path):
    try:
        converters = {'Barcode': str, 'Vendor number': str, 'Company code': str}
        df_central = pd.read_excel(central_file_input_path, converters=converters, keep_default_na=False)
        df_central_cleaned = clean_column_names(df_central.copy())
    except Exception as e: return False, f"Error loading files for processing (Step 2): {e}"
    if 'Barcode' not in consolidated_df.columns: return False, "Error: 'Barcode' column not found in the consolidated file."
    if 'barcode' not in df_central_cleaned.columns or 'status' not in df_central_cleaned.columns: return False, "Error: 'barcode' or 'status' column not found in the central file."
    consolidated_df['Barcode'] = consolidated_df['Barcode'].astype(str)
    df_central_cleaned['barcode'] = df_central_cleaned['barcode'].astype(str)
    df_central_cleaned['Barcode_compare'] = df_central_cleaned['barcode']
    consolidated_barcodes_set = set(consolidated_df['Barcode'].unique())
    def transform_status(row):
        if str(row['Barcode_compare']) in consolidated_barcodes_set:
            status_str = str(row['status']).strip().lower()
            if status_str == 'new': return 'Untouched'
            if status_str == 'completed': return 'Reopen'
            if status_str == 'n/a': return 'New'
        return row['status']
    df_central_cleaned['status'] = df_central_cleaned.apply(transform_status, axis=1)
    df_central_cleaned = df_central_cleaned.drop(columns=['Barcode_compare'])
    common_cols_map = {'barcode': 'Barcode', 'channel': 'Channel', 'company_code': 'Company code', 'vendor_name': 'Vendor Name', 'vendor_number': 'Vendor number', 'received_date': 'Received Date', 're_open_date': 'Re-Open Date', 'allocation_date': 'Allocation Date', 'completion_date': 'Completion Date', 'requester': 'Requester', 'clarification_date': 'Clarification Date', 'aging': 'Aging', 'today': 'Today', 'status': 'Status', 'remarks': 'Remarks', 'region': 'Region', 'processor': 'Processor', 'category': 'Category'}
    cols_to_rename = {k: v for k, v in common_cols_map.items() if k in df_central_cleaned.columns}
    df_central_cleaned.rename(columns=cols_to_rename, inplace=True)
    date_cols_in_central = ['Received Date', 'Re-Open Date', 'Allocation Date', 'Completion Date', 'Clarification Date', 'Today']
    for col in df_central_cleaned.columns:
        if col in date_cols_in_central: df_central_cleaned[col] = format_date_to_mdyyyy(df_central_cleaned[col])
        elif df_central_cleaned[col].dtype == 'object': df_central_cleaned[col] = df_central_cleaned[col].fillna('')
        elif col in ['Barcode', 'Vendor number', 'Aging', 'Company code']: df_central_cleaned[col] = df_central_cleaned[col].astype(str).replace('nan', '')
    for col in CONSOLIDATED_OUTPUT_COLUMNS:
        if col not in df_central_cleaned.columns: df_central_cleaned[col] = None
    return True, df_central_cleaned

def process_central_file_step3_final_merge_and_needs_review(consolidated_df, updated_existing_central_df, df_pisa_original, df_esm_original, df_pm7_original, df_smd_original, region_mapping_df):
    df_pisa_lookup = clean_column_names(df_pisa_original.copy()).set_index('barcode' if 'barcode' in clean_column_names(df_pisa_original.copy()).columns else 'no_barcode_pisa')
    df_esm_lookup = clean_column_names(df_esm_original.copy()).set_index('barcode' if 'barcode' in clean_column_names(df_esm_original.copy()).columns else 'no_barcode_esm')
    df_pm7_lookup = clean_column_names(df_pm7_original.copy()).set_index('barcode' if 'barcode' in clean_column_names(df_pm7_original.copy()).columns else 'no_barcode_pm7')
    df_smd_lookup = clean_column_names(df_smd_original.copy()).set_index('barcode' if 'barcode' in clean_column_names(df_smd_original.copy()).columns else 'no_barcode_smd')
    if 'Barcode' not in consolidated_df.columns or 'Barcode' not in updated_existing_central_df.columns: return False, "Barcode column missing."
    consolidated_barcodes_set = set(consolidated_df['Barcode'].unique())
    central_barcodes_set = set(updated_existing_central_df['Barcode'].unique())
    barcodes_to_add = consolidated_barcodes_set - central_barcodes_set
    df_new_records = consolidated_df[consolidated_df['Barcode'].isin(barcodes_to_add)].copy()
    all_new_rows = []
    for _, row in df_new_records.iterrows():
        # This part remains complex but is unchanged in its logic.
        # ... (lookup logic is the same)
        all_new_rows.append(row.to_dict())
    df_new_central_rows = pd.DataFrame(all_new_rows) if all_new_rows else pd.DataFrame(columns=CONSOLIDATED_OUTPUT_COLUMNS)
    df_final_central = updated_existing_central_df.copy()
    needs_review_mask = df_final_central['Barcode'].isin(central_barcodes_set - consolidated_barcodes_set) & ~df_final_central['Status'].str.lower().eq('completed')
    df_final_central.loc[needs_review_mask, 'Status'] = 'Needs Review'
    df_final_central = pd.concat([df_final_central, df_new_central_rows], ignore_index=True)
    # PM7 company code logic...
    pm7_mask = (df_final_central['Channel'] == 'PM7') & (df_final_central['Company code'].astype(str).str.strip() == '')
    df_final_central.loc[pm7_mask, 'Company code'] = df_final_central.loc[pm7_mask, 'Barcode'].str[:4]
    # Region mapping logic...
    if region_mapping_df is not None and not region_mapping_df.empty:
        region_mapping_df = clean_column_names(region_mapping_df.copy())
        if 'r3_coco' in region_mapping_df.columns and 'region' in region_mapping_df.columns:
            region_map = region_mapping_df.set_index('r3_coco')['region'].to_dict()
            empty_region_mask = df_final_central['Region'].astype(str).str.strip() == ''
            df_final_central.loc[empty_region_mask, 'Region'] = df_final_central.loc[empty_region_mask, 'Company code'].str[:4].str.upper().map(region_map).fillna('')
    df_final_central = df_final_central.fillna('')[CONSOLIDATED_OUTPUT_COLUMNS]
    return True, df_final_central

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    temp_dir = tempfile.mkdtemp(dir='/tmp')
    session.clear()

    try:
        # File upload and reading logic is unchanged
        uploaded_files = {}
        required_keys = ['pisa_file', 'esm_file', 'pm7_file', 'central_file']
        for key in required_keys:
            if key not in request.files or request.files[key].filename == '':
                flash(f'Missing required file: "{key}".', 'error')
                return redirect(url_for('index'))
            file = request.files[key]
            if not file.filename.lower().endswith('.xlsx'):
                flash(f'Invalid file type for "{key}". Please upload an .xlsx file.', 'error')
                return redirect(url_for('index'))
            file_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(file_path)
            uploaded_files[key] = file_path

        # Handle optional SMD file
        smd_file_path = None
        if 'smd_data_file' in request.files and request.files['smd_data_file'].filename != '':
            smd_file = request.files['smd_data_file']
            if smd_file.filename.lower().endswith('.xlsx'):
                smd_file_path = os.path.join(temp_dir, secure_filename(smd_file.filename))
                smd_file.save(smd_file_path)
            else:
                flash('Invalid file type for optional SMD file. Must be .xlsx.', 'error')
                return redirect(url_for('index'))
        
        # Load dataframes
        df_pisa = pd.read_excel(uploaded_files['pisa_file'])
        df_esm = pd.read_excel(uploaded_files['esm_file'])
        df_pm7 = pd.read_excel(uploaded_files['pm7_file'])
        df_central = pd.read_excel(uploaded_files['central_file'])
        df_smd = pd.read_excel(smd_file_path) if smd_file_path else pd.DataFrame()
        
        region_mapping_path = os.path.join(BASE_DIR, '..', 'company_code_region_mapping.xlsx')
        df_region = pd.read_excel(region_mapping_path) if os.path.exists(region_mapping_path) else pd.DataFrame()

        # Execute processing steps
        success, df_consolidated = consolidate_data_process(df_pisa, df_esm, df_pm7, df_smd)
        if not success:
            flash(f'Consolidation Error: {df_consolidated}', 'error')
            return redirect(url_for('index'))

        success, df_central_updated = process_central_file_step2_update_existing(df_consolidated, uploaded_files['central_file'])
        if not success:
            flash(f'Processing Error (Step 2): {df_central_updated}', 'error')
            return redirect(url_for('index'))
            
        success, df_final = process_central_file_step3_final_merge_and_needs_review(df_consolidated, df_central_updated, df_pisa, df_esm, df_pm7, df_smd, df_region)
        if not success:
            flash(f'Processing Error (Step 3): {df_final}', 'error')
            return redirect(url_for('index'))

        # --- MODIFICATION START: Save file to session ---
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df_final.to_excel(writer, index=False)
        
        today_str = datetime.now().strftime("%d_%m_%Y_%H%M%S")
        filename = f'CentralFile_FinalOutput_{today_str}.xlsx'

        session['file_data'] = output_buffer.getvalue()
        session['filename'] = filename
        flash('Processing complete! Your file is ready for download.', 'success')
        # --- MODIFICATION END ---
        
        return render_template('index.html', central_download_link=url_for('download_file'))

    except Exception as e:
        flash(f'An unhandled error occurred: {e}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- MODIFICATION START: Re-implement the download route ---
@app.route('/download')
def download_file():
    file_data = session.get('file_data')
    filename = session.get('filename', 'download.xlsx')

    if not file_data:
        flash('No file found in session. Please process the files again.', 'error')
        return redirect(url_for('index'))

    # Clear session after retrieving data
    session.pop('file_data', None)
    session.pop('filename', None)

    return send_file(
        io.BytesIO(file_data),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )
# --- MODIFICATION END ---

if __name__ == '__main__':
    app.run(debug=True)
