from __future__ import annotations

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd

from loader import load_all_events, sort_session_events, get_timestamp_ms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'raw_data' / 'dataset-downloads'
SEGMENTS_FILE = PROJECT_ROOT / 'outputs' / 'dataset_b_segments.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
SEGMENT_INVENTORY_FILE = OUTPUT_DIR / 'dataset_b_segment_inventory.csv'
PROCESS_FAMILY_FILE = OUTPUT_DIR / 'dataset_b_process_families.csv'

MAX_APP_SEQUENCE_LENGTH = 8
SHORT_SEGMENT_SECONDS = 5
DURATION_BINS = [0, 10, 30, 60, 120, 300, 600, np.inf]
DURATION_LABELS = ['<10s', '10-30s', '30-60s', '1-2m', '2-5m', '5-10m', '10m+']


def get_event_type(event):
    value = event.get('event_type')
    return value if isinstance(value, str) else ''


def get_event_app(event):
    context = event.get('context') or {}
    active_app = context.get('active_app') or {}
    app_name = active_app.get('app_name')
    return app_name.strip() if isinstance(app_name, str) and app_name.strip() else None


def get_context_extracted_text(event):
    context = event.get('context') or {}
    text = context.get('extracted_text')
    return text.strip() if isinstance(text, str) and text.strip() else None


def get_browser_url(event):
    payload = event.get('payload') or {}
    context = event.get('context') or {}
    candidates = [payload.get('url'), payload.get('current_url'), payload.get('target_url'),
                  context.get('url'), context.get('current_url'), context.get('browser_url')]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_browser_domain(url):
    if not url:
        return None
    value = re.sub(r'^[a-z]+://', '', url.strip().lower())
    value = value.split('/', 1)[0].split(':', 1)[0]
    return value[4:] if value.startswith('www.') else value


def normalize_app_name(app_name):
    if not isinstance(app_name, str) or not app_name.strip():
        return 'UNKNOWN'
    return re.sub(r'\s+', ' ', app_name.strip())


def collapse_consecutive(values):
    output = []
    for value in values:
        if value and (not output or output[-1] != value):
            output.append(value)
    return output


def truncate_sequence(values, max_length):
    if len(values) <= max_length:
        return values
    left = max_length // 2
    right = max_length - left
    return values[:left] + ['...'] + values[-right:]


def prepare_session_events(events):
    prepared = []
    for event in sort_session_events(events):
        timestamp_ms = get_timestamp_ms(event)
        if timestamp_ms is None:
            continue
        prepared.append({
            'timestamp_ms': int(timestamp_ms),
            'event_type': get_event_type(event),
            'app': normalize_app_name(get_event_app(event)),
            'extracted_text': get_context_extracted_text(event),
            'browser_url': get_browser_url(event),
        })
    return prepared


def events_in_segment(prepared_events, start, end):
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return [e for e in prepared_events if start_ms <= e['timestamp_ms'] <= end_ms]


def build_segment_record(segment_row, prepared_events):
    session_id = segment_row['session_id']
    segment_index = int(segment_row['segment_index'])
    start = pd.Timestamp(segment_row['start'])
    end = pd.Timestamp(segment_row['end'])
    duration = float(segment_row['duration_seconds'])
    segment_events = events_in_segment(prepared_events, start, end)

    event_types = [e['event_type'] for e in segment_events if e['event_type']]
    apps = [e['app'] for e in segment_events if e['app'] != 'UNKNOWN']
    collapsed_apps = truncate_sequence(collapse_consecutive(apps), MAX_APP_SEQUENCE_LENGTH)
    app_counts = Counter(apps)
    event_counts = Counter(event_types)
    extracted_text_values = [e['extracted_text'] for e in segment_events if e['extracted_text']]
    urls = [e['browser_url'] for e in segment_events if e['browser_url']]
    domains = truncate_sequence(collapse_consecutive([get_browser_domain(u) for u in urls if get_browser_domain(u)]), MAX_APP_SEQUENCE_LENGTH)
    unique_apps = sorted(set(apps))
    unique_event_types = sorted(set(event_types))
    duration_bucket = pd.cut([duration], bins=DURATION_BINS, labels=DURATION_LABELS, include_lowest=True, right=False)[0]
    text_sample = ' | '.join(extracted_text_values[:5])
    if len(text_sample) > 500:
        text_sample = text_sample[:497] + '...'

    app_sequence = ' > '.join(collapsed_apps)
    domain_sequence = ' > '.join(domains)
    event_type_sequence = ' > '.join(truncate_sequence(collapse_consecutive(event_types), MAX_APP_SEQUENCE_LENGTH * 2))

    record = {
        'session_id': session_id,
        'segment_index': segment_index,
        'start': start,
        'end': end,
        'duration_seconds': duration,
        'duration_bucket': duration_bucket,
        'event_count': len(segment_events),
        'dominant_app': app_counts.most_common(1)[0][0] if app_counts else 'UNKNOWN',
        'unique_app_count': len(unique_apps),
        'apps': ' | '.join(unique_apps),
        'app_sequence': app_sequence,
        'browser_domain_sequence': domain_sequence,
        'unique_event_type_count': len(unique_event_types),
        'dominant_event_type': event_counts.most_common(1)[0][0] if event_counts else 'UNKNOWN',
        'event_type_sequence': event_type_sequence,
        'app_switch_count': event_counts['app_switch'],
        'browser_navigation_count': event_counts['browser_navigation'],
        'browser_click_count': event_counts['browser_click'],
        'browser_form_input_count': event_counts['browser_form_input'],
        'mouse_click_count': event_counts['mouse_click'],
        'keyboard_count': event_counts['keystroke'] + event_counts['text_input_complete'],
        'clipboard_change_count': event_counts['clipboard_change'],
        'extracted_text_event_count': len(extracted_text_values),
        'extracted_text_available': bool(extracted_text_values),
        'extracted_text_sample': text_sample,
        'short_segment_flag': duration < SHORT_SEGMENT_SECONDS,
    }
    record['process_signature'] = (
        f"apps={app_sequence or 'NONE'} | duration={duration_bucket} | "
        f"nav={record['browser_navigation_count']} | form={record['browser_form_input_count']}"
    )
    return record


def build_family_inventory(inventory):
    df = inventory.copy()
    df['family_key'] = (
        df['app_sequence'].fillna('') + ' || ' + df['duration_bucket'].astype(str) +
        ' || nav=' + df['browser_navigation_count'].astype(str) +
        ' || form=' + df['browser_form_input_count'].astype(str)
    )
    families = (df.groupby('family_key').agg(
        segment_count=('segment_index', 'size'),
        session_count=('session_id', 'nunique'),
        total_duration_seconds=('duration_seconds', 'sum'),
        median_duration_seconds=('duration_seconds', 'median'),
        mean_duration_seconds=('duration_seconds', 'mean'),
        min_duration_seconds=('duration_seconds', 'min'),
        max_duration_seconds=('duration_seconds', 'max'),
        dominant_app=('dominant_app', lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN'),
        extracted_text_coverage=('extracted_text_available', 'mean'),
    ).reset_index().sort_values(['segment_count', 'total_duration_seconds'], ascending=[False, False]).reset_index(drop=True))
    families['family_id'] = [f'F{i:03d}' for i in range(1, len(families) + 1)]
    family_map = dict(zip(families['family_key'], families['family_id']))
    df['family_id'] = df['family_key'].map(family_map)
    df = df.drop(columns=['family_key'])
    return df, families[['family_id', 'family_key', 'segment_count', 'session_count', 'total_duration_seconds', 'median_duration_seconds', 'mean_duration_seconds', 'min_duration_seconds', 'max_duration_seconds', 'dominant_app', 'extracted_text_coverage']]


def main():
    print('=' * 70)
    print('DATASET B SEGMENT INVENTORY')
    print('=' * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(f'Dataset B segment file not found:\n{SEGMENTS_FILE}')
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f'Raw data root not found:\n{DATA_ROOT}')

    segments = pd.read_csv(SEGMENTS_FILE)
    required = {'session_id', 'segment_index', 'start', 'end', 'duration_seconds'}
    missing = required - set(segments.columns)
    if missing:
        raise RuntimeError('Dataset B segment file is missing required columns:\n' + '\n'.join(sorted(missing)))
    segments['start'] = pd.to_datetime(segments['start'], utc=True)
    segments['end'] = pd.to_datetime(segments['end'], utc=True)

    print('\nLoading raw Dataset B sessions...')
    all_sessions = load_all_events(DATA_ROOT)
    dataset_b_sessions = {}
    for session_id, events in all_sessions.items():
        datasets = {e.get('_dataset') for e in events if e.get('_dataset')}
        if 'B' in datasets:
            dataset_b_sessions[session_id] = events
    print(f'Dataset B sessions found: {len(dataset_b_sessions):,}')

    expected_sessions = set(segments['session_id'])
    missing_sessions = expected_sessions - set(dataset_b_sessions)
    if missing_sessions:
        raise RuntimeError('Some segmented sessions were not found in raw Dataset B:\n' + '\n'.join(sorted(missing_sessions)))

    prepared_by_session = {sid: prepare_session_events(events) for sid, events in dataset_b_sessions.items()}

    print('\nBuilding segment-level signatures...')
    records = []
    grouped_segments = segments.sort_values(['session_id', 'segment_index']).groupby('session_id')
    total_groups = segments['session_id'].nunique()
    for processed, (session_id, session_segments) in enumerate(grouped_segments, start=1):
        prepared = prepared_by_session[session_id]
        for _, segment_row in session_segments.iterrows():
            records.append(build_segment_record(segment_row, prepared))
        if processed % 5 == 0 or processed == total_groups:
            print(f'Processed {processed}/{total_groups} sessions')

    inventory = pd.DataFrame(records)
    if inventory.empty:
        raise RuntimeError('No segment inventory was generated.')

    print('\nInventory summary:')
    print(f"Segments: {len(inventory):,}")
    print(f"Sessions: {inventory['session_id'].nunique():,}")
    print(f"Mean duration: {inventory['duration_seconds'].mean():.2f}s")
    print(f"Median duration: {inventory['duration_seconds'].median():.2f}s")
    print(f"Extracted-text coverage: {inventory['extracted_text_available'].mean():.2%}")

    inventory, families = build_family_inventory(inventory)
    inventory.to_csv(SEGMENT_INVENTORY_FILE, index=False, encoding='utf-8-sig')
    families.to_csv(PROCESS_FAMILY_FILE, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 70)
    print('TOP PROCESS-FAMILY HYPOTHESES')
    print('=' * 70)
    print(families[['family_id', 'segment_count', 'session_count', 'total_duration_seconds', 'median_duration_seconds', 'dominant_app']].head(20).to_string(index=False))

    print('\n' + '=' * 70)
    print('TOP APP-SEQUENCE PATTERNS')
    print('=' * 70)
    app_patterns = (inventory.groupby(['app_sequence', 'duration_bucket']).agg(
        segment_count=('segment_index', 'size'), session_count=('session_id', 'nunique'),
        total_duration_seconds=('duration_seconds', 'sum'), median_duration_seconds=('duration_seconds', 'median')
    ).reset_index().sort_values(['segment_count', 'total_duration_seconds'], ascending=[False, False]).head(30))
    print(app_patterns.to_string(index=False))

    print('\n' + '=' * 70)
    print('TOP DOMINANT APPS')
    print('=' * 70)
    dominant_apps = (inventory.groupby('dominant_app').agg(
        segment_count=('segment_index', 'size'), session_count=('session_id', 'nunique'),
        total_duration_seconds=('duration_seconds', 'sum')
    ).reset_index().sort_values(['segment_count', 'total_duration_seconds'], ascending=[False, False]).head(20))
    print(dominant_apps.to_string(index=False))

    print('\n' + '=' * 70)
    print('DATASET B INVENTORY COMPLETE')
    print('=' * 70)
    print(f'Segment inventory:\n{SEGMENT_INVENTORY_FILE}')
    print(f'\nProcess-family hypotheses:\n{PROCESS_FAMILY_FILE}')


if __name__ == '__main__':
    main()
