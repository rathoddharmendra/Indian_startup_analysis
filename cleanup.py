import os, re, io
import pandas as pd
from collections import OrderedDict

root = '/Users/mac_dee/Documents/Dee/code/data_analytics_projects/Indian_startup_analysis'
raw_path = os.path.join(root, 'raw', 'startup_funding.csv')
clean_dir = os.path.join(root, 'cleaned')
os.makedirs(clean_dir, exist_ok=True)
clean_csv_path = os.path.join(clean_dir, 'startup_funding_clean.csv')
md_path = os.path.join(root, 'analysis.md')

with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()


text = text.replace('\r\n', '\n').replace('\r', '\n')


text_one_line = re.sub(r'\n(?!\s*\d+,)', ' ', text)

lines = [ln for ln in text_one_line.split('\n') if ln.strip()]
if not lines:
    raise SystemExit('Raw CSV seems empty after preprocessing')

header_parts = []
record_lines = []
header_done = False
for ln in lines:
    if not header_done and re.match(r'^\s*\d+\s*,', ln):
        header_done = True
        record_lines.append(ln)
    elif not header_done:
        header_parts.append(ln)
    else:
        record_lines.append(ln)

header = ' '.join(header_parts).strip()

header = re.sub(r'\s{2,}', ' ', header)

cleaned_text = header + '\n' + '\n'.join(record_lines) + '\n'

from io import StringIO
try:
    df = pd.read_csv(StringIO(cleaned_text), engine='python')
except Exception as e:
    df = pd.read_csv(StringIO(cleaned_text), engine='python', on_bad_lines='skip')

original_cols = list(df.columns)
normalized = []
for col in original_cols:
    c = str(col).strip()
    c = re.sub(r'\s{2,}', ' ', c)
    c = c.replace('dd/mm/yyyy', '').strip()
    c = c.replace('  ', ' ')
    c = c.replace('–', '-')
    normalized.append(c)

df.columns = normalized

colmap = {}
cols_lower = {c.lower(): c for c in df.columns}

def pick(keys):
    for k in keys:
        for c in df.columns:
            if k in c.lower():
                return c
    return None

colmap['sr_no'] = pick(['sr no', 's.no', 'sr', 's no'])
colmap['date'] = pick(['date'])
colmap['startup'] = pick(['startup name', 'startup'])
colmap['industry'] = pick(['industry vertical', 'industry'])
colmap['subvertical'] = pick(['subvertical', 'sub vertical', 'sub-vertical'])
colmap['city'] = pick(['city', 'location'])
colmap['investors'] = pick(['investors name', 'investor', 'investors'])
colmap['investment_type'] = pick(['investment type', 'investmentntype', 'investmentn', 'round'])
colmap['amount_usd'] = pick(['amount in usd', 'amount usd', 'amount'])
colmap['remarks'] = pick(['remarks', 'comment', 'notes'])

selected_cols = [v for v in colmap.values() if v is not None]
rename_map = {v: k for k, v in colmap.items() if v is not None}

df = df[selected_cols].rename(columns=rename_map)

if 'sr_no' in df.columns:
    df['sr_no'] = pd.to_numeric(df['sr_no'], errors='coerce')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['year'] = df['date'].dt.year

if 'city' in df.columns:
    df['city'] = df['city'].astype(str).str.strip()
    df['city'] = df['city'].str.replace('\s+', ' ', regex=True)

if 'investors' in df.columns:
    df['investors'] = df['investors'].astype(str).str.replace('\sand\s', ',', regex=True)
    df['investors'] = df['investors'].str.replace(';', ',', regex=False)
    df['investors'] = df['investors'].str.replace('&', ',', regex=False)
    df['investors'] = df['investors'].str.replace('/', ',', regex=False)
    df['investors'] = df['investors'].str.replace(',,', ',', regex=False)
    df['investors'] = df['investors'].str.strip(' ,')

if 'investment_type' in df.columns:
    df['investment_type'] = df['investment_type'].astype(str).str.strip()

import numpy as np
if 'amount_usd' in df.columns:
    amt = df['amount_usd'].astype(str)
    amt = amt.str.replace('[^0-9.]', '', regex=True)
    # If empty -> NaN
    amt = amt.replace({'': np.nan})
    df['amount_usd_num'] = pd.to_numeric(amt, errors='coerce')

if 'startup' in df.columns:
    df['startup'] = df['startup'].astype(str).str.strip()
    df = df[df['startup'].astype(str).str.len() > 0]

df.to_csv(clean_csv_path, index=False)

lines = []
lines.append('# Indian Startup Funding Analysis')
lines.append('')
lines.append(f'- Records: {len(df)}')
if 'date' in df.columns:
    valid_dates = df['date'].dropna()
    if not valid_dates.empty:
        lines.append(f'- Date range: {valid_dates.min().date()} to {valid_dates.max().date()}')
if 'amount_usd_num' in df.columns:
    total_amt = df['amount_usd_num'].sum(skipna=True)
    lines.append(f'- Total disclosed amount (USD, parsed numerically): {int(total_amt):,}')
lines.append('')

# Helper to render markdown tables without external deps

def to_md_table(frame, index=False, max_rows=15):
    f = frame.copy()
    if not index:
        f = f.reset_index(drop=True)
    # limit rows
    if len(f) > max_rows:
        f = f.head(max_rows)
    cols = list(f.columns)
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body_lines = []
    for _, row in f.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if abs(v) >= 1_000_000:
                    vals.append(f"{v:,.0f}")
                elif abs(v) >= 1_000:
                    vals.append(f"{v:,.0f}")
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        body_lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join([header, sep] + body_lines)

# Funding by year
if {'year','amount_usd_num'}.issubset(df.columns):
    by_year = df.groupby('year', dropna=True)['amount_usd_num'].sum().reset_index().sort_values('year')
    by_year['amount_usd_num'] = by_year['amount_usd_num'].round(0).astype('int64')
    by_year = by_year.rename(columns={'amount_usd_num':'total_amount_usd'})
    lines.append('## Total funding by year')
    lines.append('')
    lines.append(to_md_table(by_year))
    lines.append('')

# Deals by city
if 'city' in df.columns:
    deals_city = df['city'].value_counts().reset_index()
    deals_city.columns = ['city','deal_count']
    lines.append('## Top cities by number of deals')
    lines.append('')
    lines.append(to_md_table(deals_city.head(15)))
    lines.append('')

# Top sectors by total amount
if {'industry','amount_usd_num'}.issubset(df.columns):
    by_sector = df.groupby('industry')['amount_usd_num'].sum().reset_index().sort_values('amount_usd_num', ascending=False)
    by_sector['amount_usd_num'] = by_sector['amount_usd_num'].round(0).astype('int64')
    by_sector = by_sector.rename(columns={'industry':'industry_vertical','amount_usd_num':'total_amount_usd'})
    lines.append('## Top industries by total funding')
    lines.append('')
    lines.append(to_md_table(by_sector.head(15)))
    lines.append('')

# Top startups by total amount
if {'startup','amount_usd_num'}.issubset(df.columns):
    by_startup = df.groupby('startup')['amount_usd_num'].sum().reset_index().sort_values('amount_usd_num', ascending=False)
    by_startup['amount_usd_num'] = by_startup['amount_usd_num'].round(0).astype('int64')
    by_startup = by_startup.rename(columns={'amount_usd_num':'total_amount_usd'})
    lines.append('## Top startups by total funding')
    lines.append('')
    lines.append(to_md_table(by_startup.head(15)))
    lines.append('')

if 'investors' in df.columns:
    inv = df[['investors']].copy()
    inv['investors'] = inv['investors'].fillna('')
    inv = inv[inv['investors'].str.len() > 0]
    inv = inv.assign(investor=inv['investors'].str.split(',')).explode('investor')
    inv['investor'] = inv['investor'].str.strip()
    inv = inv[inv['investor'].str.len() > 0]
    top_investors = inv['investor'].value_counts().reset_index().rename(columns={'index':'investor','investor':'deal_count'})
    lines.append('## Top investors by number of deals')
    lines.append('')
    lines.append(to_md_table(top_investors.head(15)))
    lines.append('')

with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('CLEAN_CSV:', clean_csv_path)
print('MARKDOWN:', md_path)
