"""Candidate likelihood and unedited deterministic generation, one image at a time."""
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .io import atomic_json, digest, now, write_jsonl
from .models.answer_scoring import parse_answer, restricted_probabilities
from .statistics import summarize


def image_for(row):
    if row.get('question_roi_path_or_null'):
        raise ValueError('whole-image core rejects an unaudited question ROI')
    with Image.open(row['image_path']) as image:
        return image.convert('RGB')


def evaluate_model(model, rows, output, free_generation=True):
    if not rows:
        raise ValueError('empty admitted evaluation role')
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite prediction artifacts')
    model.eval()
    predictions, started = [], time.monotonic()
    for index, row in enumerate(rows):
        image = image_for(row)
        candidates = row['answer_candidates']
        scores = model.score_candidates([image], [row['question']], [candidates])[0]
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise FloatingPointError('candidate scores are not finite and aligned')
        record = {key: row[key] for key in ('qa_id', 'canonical_image_id', 'group_id', 'source_dataset',
                                           'split', 'question_type', 'answer', 'answer_candidates')}
        record.update(annotation_complete=row['annotation_complete'], answer_provenance=row['answer_provenance'])
        if 'lesion_type' in row:
            record['lesion_type'] = row['lesion_type']
        record.update(prediction=candidates[int(np.argmax(scores))], scores=scores,
                      candidate_probabilities=restricted_probabilities(scores), question=row['question'])
        if free_generation:
            generated = model.generate_answers([image], [row['question']])[0]
            record.update(generation=generated, parsed_generation=parse_answer(generated, candidates))
        predictions.append(record)
        if index % 20 == 0:
            atomic_json(output.with_suffix('.progress.json'),
                        {'completed': index + 1, 'total': len(rows), 'updated_at': now()})
    write_jsonl(output, predictions)
    metrics = summarize(predictions)
    atomic_json(output.with_suffix('.metrics.json'), metrics)
    return {'rows': len(predictions), 'sha256': digest(output), 'elapsed_seconds': time.monotonic() - started,
            'metrics': metrics}
