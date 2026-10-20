"""Join explicitly typed lesion QA to verified original images; no text guessing."""
from .lesion_statistics import LESIONS


def assemble_lesions(rows, mapping):
    types = mapping.get('question_type_to_lesion')
    if (not isinstance(types, dict) or len(types) != len(LESIONS) or
            set(types.values()) != set(LESIONS) or any(not isinstance(key,str) or not key for key in types)):
        raise ValueError('mapping must declare exactly one unique question type for each EX/HE/MA/SE lesion')
    if not rows:
        raise ValueError('nonempty lesion QA predictions required')
    grouped, qa_ids = {}, set()
    for row in rows:
        required = ('qa_id','canonical_image_id','group_id','question_type','answer','answer_candidates','prediction','annotation_complete')
        if set(required) - set(row):
            raise ValueError('lesion QA prediction is missing required identity/reference fields')
        if row['qa_id'] in qa_ids:
            raise ValueError('duplicate QA identity in lesion assembly')
        qa_ids.add(row['qa_id'])
        if row['question_type'] not in types:
            raise ValueError('question type is absent from the explicit lesion mapping')
        lesion = types[row['question_type']]
        if row.get('lesion_type',lesion) != lesion:
            raise ValueError('explicit lesion_type disagrees with the question-type mapping')
        candidates = row['answer_candidates']
        if (not isinstance(candidates,list) or len(candidates)!=2 or set(candidates)!={'no','yes'} or
                row['answer'] not in ('no','yes') or row['prediction'] not in ('no','yes',None)):
            raise ValueError('lesion references/candidates must be yes/no; prediction must be yes/no/null')
        if row['annotation_complete'] is not True:
            raise ValueError('incomplete lesion channel cannot be assumed negative')
        image_id, group = row['canonical_image_id'], row['group_id']
        if not isinstance(image_id,str) or not image_id or not isinstance(group,str) or not group:
            raise ValueError('nonempty original-image and group identities required')
        role = row.get('role',row.get('split'))
        if role not in ('train','val','test') or row.get('split',role)!=role:
            raise ValueError('lesion QA must have a consistent explicit role/split')
        if image_id not in grouped:
            grouped[image_id] = dict(sample_id=image_id,canonical_image_id=image_id,group_id=group,role=role,
                                     evidence_truth={},lesions={})
        merged = grouped[image_id]
        if merged['group_id'] != group or merged['role'] != role:
            raise ValueError('one original image has inconsistent group/role identities')
        if lesion in merged['lesions']:
            raise ValueError('duplicate lesion question for the same original image')
        merged['evidence_truth'][lesion] = row['answer']=='yes'
        detail = {key:row[key] for key in ('prediction','qa_id','question_type')}
        for key in ('parsed_generation','generation','scores','candidate_probabilities','answer_provenance'):
            if key in row:
                detail[key] = row[key]
        merged['lesions'][lesion] = detail
    for image in grouped.values():
        if set(image['evidence_truth']) != set(LESIONS):
            raise ValueError('all four explicitly annotated lesion channels are required for every image')
    return [grouped[key] for key in sorted(grouped)]
