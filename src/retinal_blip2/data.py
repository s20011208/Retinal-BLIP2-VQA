"""Identity first, role before answer decoding, original-group-only budgets.

Identity manifests contain no answers. Canonical answer JSONL must contain one
literal top-level role marker; escaped schema keys and duplicate markers are
rejected before decoding. This intentionally replaces the historical mixed-file
reader implicated in I024; it does not undo that historical exposure.
"""
import json
import hashlib
import math
from pathlib import Path
import random
import re

from .io import digest

ROLES = ('train', 'val', 'test')
IDENTITY_REQUIRED = ('qa_id', 'canonical_image_id', 'group_id', 'role', 'image_path', 'source_dataset')
ANSWER_REQUIRED = ('qa_id', 'role', 'question', 'question_type', 'answer_candidates', 'answer',
                   'answer_provenance', 'annotation_complete')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def identities(path, data_root=None, check_images=True):
    result, group_roles, image_roles, raw_roles, pixel_roles, patient_roles = {}, {}, {}, {}, {}, {}
    root = Path(data_root or '.')
    content_cache, actual_raw_roles, actual_rgb_roles = {}, {}, {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique_object)
            if set(IDENTITY_REQUIRED) - set(row):
                raise ValueError('identity missing fields on line %d' % number)
            if any(key in row for key in ('answer', 'label', 'labels', 'evidence_truth', 'masks')):
                raise PermissionError('identity manifest must not contain reference annotations')
            if row['qa_id'] in result or row['role'] not in ROLES:
                raise ValueError('duplicate QA identity or unknown role')
            for key in ('qa_id', 'canonical_image_id', 'group_id', 'source_dataset', 'image_path'):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError('empty or invalid identity: ' + key)
            for value, mapping, field in ((row['group_id'], group_roles, 'group'),
                                           (row['canonical_image_id'], image_roles, 'image'),
                                           (row.get('raw_sha256'), raw_roles, 'raw bytes'),
                                           (row.get('pixel_sha256'), pixel_roles, 'decoded pixels'),
                                           (row.get('patient_id'), patient_roles, 'patient')):
                if value is None:
                    continue
                if value in mapping and mapping[value] != row['role']:
                    raise ValueError('cross-role overlap in ' + field)
                mapping[value] = row['role']
            image = Path(row['image_path'])
            row['image_path'] = str(image if image.is_absolute() else root / image)
            if check_images and not Path(row['image_path']).is_file():
                raise FileNotFoundError('identity image does not exist: ' + row['canonical_image_id'])
            if check_images:
                from PIL import Image
                actual_path = str(Path(row['image_path']).resolve())
                if actual_path not in content_cache:
                    raw_hash = digest(actual_path)
                    with Image.open(actual_path) as source:
                        rgb = source.convert('RGB')
                        pixel_hash = hashlib.sha256(('%dx%d:' % rgb.size).encode() + rgb.tobytes()).hexdigest()
                    content_cache[actual_path] = (raw_hash,pixel_hash)
                raw_hash,pixel_hash = content_cache[actual_path]
                if row.get('raw_sha256') and row['raw_sha256'] != raw_hash:
                    raise ValueError('identity raw hash does not match actual image bytes')
                for content,mapping,label in ((raw_hash,actual_raw_roles,'actual raw image bytes'),
                                               (pixel_hash,actual_rgb_roles,'actual decoded RGB pixels')):
                    if content in mapping and mapping[content] != row['role']:
                        raise ValueError('cross-role overlap in ' + label)
                    mapping[content] = row['role']
            if row.get('question_roi_path_or_null'):

                raise ValueError('whole-image core does not accept unaudited question ROI')
            result[row['qa_id']] = row
    if not result:
        raise ValueError('empty identity manifest')
    return result


def _literal_marker(raw, key):
    keys = re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', raw)
    if any('\\' in item for item in keys):
        raise PermissionError('escaped schema key rejected before answer decoding')
    if keys.count(key) != 1:
        raise PermissionError('missing/duplicate literal ' + key + ' key rejected before answer decoding')
    values = re.findall(r'(?<!\\)"' + key + r'"\s*:\s*"([^"\\]*)"', raw)
    if len(values) != 1:
        raise PermissionError('missing/ambiguous literal ' + key + ' rejected before answer decoding')
    return values[0]


def load_qa(identity_path, answer_path, roles=('train', 'val'), data_root=None,
            check_images=True, receipt_path=None):
    roles = tuple(roles)
    if not roles or any(role not in ROLES for role in roles):
        raise ValueError('explicit valid data roles required')
    identity_hash, answer_hash = digest(identity_path), digest(answer_path)
    known = identities(identity_path, data_root, check_images)
    if 'test' in roles:
        if receipt_path is None:
            raise PermissionError('test answers require a local frozen-study receipt')
        receipt = json.loads(Path(receipt_path).read_text())
        if (receipt.get('status') != 'FROZEN_FOR_EVALUATION' or
                receipt.get('identities_sha256') != identity_hash or
                receipt.get('answers_sha256') != answer_hash):
            raise PermissionError('test receipt does not bind these identity/answer bytes')
        if not check_images:
            raise PermissionError('test roles require verification of actual image bytes')
        if any(not row.get('raw_sha256') for row in known.values() if row['role']=='test'):
            raise PermissionError('every test image needs a frozen raw_sha256 identity binding')
    allowed = {key: value for key, value in known.items() if value['role'] in roles}
    rows, seen, excluded = [], set(), 0
    with Path(answer_path).open() as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            role = _literal_marker(raw, 'role')
            if role not in ROLES:
                raise PermissionError('unknown role before answer decoding')
            if role not in roles:
                excluded += 1
                continue
            key = _literal_marker(raw, 'qa_id')
            if key not in allowed:
                excluded += 1
                continue
            if role != allowed[key]['role']:
                raise PermissionError('answer role differs from admitted identity before decoding')
            row = json.loads(raw, object_pairs_hook=unique_object)
            if set(ANSWER_REQUIRED) - set(row):
                raise ValueError('answer schema missing fields on admitted line %d' % number)
            if row['qa_id'] in seen:
                raise ValueError('duplicate admitted QA answer')
            if type(row.get('ambiguity_flag', False)) is not bool:
                raise ValueError('ambiguity_flag must be a boolean')
            if (row['annotation_complete'] is not True or row.get('ambiguity_flag', False) or
                    row.get('question_roi_path_or_null')):
                raise ValueError('incomplete, ambiguous or regional annotations are outside the whole-image core')
            candidates = row['answer_candidates']
            if (not isinstance(candidates, list) or not candidates or
                    any(not isinstance(item, str) or not item for item in candidates) or
                    len(set(candidates)) != len(candidates) or row['answer'] not in candidates):
                raise ValueError('reference answer outside unique frozen candidate set')
            for field in ('question', 'question_type', 'answer_provenance'):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError('nonempty string required: ' + field)
            merged = dict(row)
            merged.update(allowed[key])
            merged['split'] = role
            rows.append(merged)
            seen.add(key)
    if seen != set(allowed):
        raise ValueError('some admitted identities lack exactly one complete QA answer')
    if digest(identity_path) != identity_hash or digest(answer_path) != answer_hash:
        raise RuntimeError('dataset changed during admission')
    return rows, {'identities_sha256': identity_hash, 'answers_sha256': answer_hash,
                  'admitted_qa': len(rows), 'excluded_before_answer_decode': excluded,
                  'roles': list(roles), 'original_groups': len({r['group_id'] for r in rows}),
                  'patient_ids_unknown': any(not r.get('patient_id') for r in rows),
                  'overlap_scope': 'declared group/image/patient IDs and supplied hashes; actual raw/RGB image hashes when image checks enabled; not near-duplicate certification'}


def grouped_budget(rows, fraction, seed):
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError('budget must be in (0,1]')
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('budget selection requires admitted training-only rows')
    groups = sorted({row['group_id'] for row in rows})
    random.Random(seed).shuffle(groups)
    required = {(row['question_type'], row['answer']) for row in rows}
    by_group = {group: [] for group in groups}
    for row in rows:
        by_group[row['group_id']].append(row)
    covered, minimum = set(), 0
    for group in groups:
        minimum += 1
        covered.update((row['question_type'], row['answer']) for row in by_group[group])
        if covered == required:
            break
    target = max(1, math.ceil(len(groups) * fraction))
    count = max(target, minimum)
    selected = set(groups[:count])
    chosen = [row for row in rows if row['group_id'] in selected]
    return chosen, {'requested_fraction': fraction, 'seed': seed, 'available_groups': len(groups),
                    'actual_groups': count, 'actual_images': len({row['canonical_image_id'] for row in chosen}),
                    'qa_rows': len(chosen), 'groups': sorted(selected),
                    'minimum_class_covering_prefix': minimum, 'raised_for_training_class_coverage': count > target}
