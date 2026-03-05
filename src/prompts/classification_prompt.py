CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE = """You assign customer questions to exactly one cluster each.

Clusters (use these ids in your response):
{clusters_text}

For each question, return the cluster id that best matches it. Use only the question text and cluster descriptions.
Every question must be assigned to exactly one cluster.
There are no exceptions: map every question to one of the clusters listed above.
If a question is ambiguous or only loosely related, choose the cluster that fits best.
Never leave any question unassigned.

Respond with a single valid JSON object (no markdown, no extra text) in this exact format:
{{"assignments": [{{"question_id": 0, "cluster_id": "C01"}}, {{"question_id": 1, "cluster_id": "C02"}}, ...]}}

question_id is the 0-based index of the question in the list you receive. cluster_id is a string and must be one of the cluster ids listed above (e.g. C01, C02).
You must provide an assignment for every question."""


CLASSIFICATION_USER_PROMPT_TEMPLATE = """Questions to assign (question_id is the index in the list below):

{questions_text}"""
