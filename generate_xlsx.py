import json, openpyxl
from src.pipeline import run_pipeline

jd = "Senior Backend Engineer (5+ years) needed. Must have Python, PostgreSQL, AWS, and Docker. Microservices experience required. Nice to have: Kubernetes, Kafka, Terraform."

result = run_pipeline(jd, 'data/sample_candidates.json', top_k=16)

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Ranked Candidates'
ws.append(['Rank', 'Name', 'Final Score', 'Semantic Fit', 'Hard Match', 'Career Fit', 'Behavioral', 'Matched Skills', 'Missing Skills', 'Quality Flags'])

for r in result['shortlist']:
    c = r['candidate']
    bd = r['score_breakdown']
    hd = r['hard_requirement_details']
    ws.append([
        r['rank'],
        c['name'],
        r['final_score'],
        bd['semantic_fit'],
        bd['hard_requirement_fit'],
        bd['career_metadata_fit'],
        bd['behavioral_signal'],
        ', '.join(hd['matched_must_have']),
        ', '.join(hd['missing_must_have']),
        ', '.join(r['quality_flags']) if r['quality_flags'] else 'None'
    ])

wb.save('ranked_candidates_output.xlsx')
print('Done! File saved.')