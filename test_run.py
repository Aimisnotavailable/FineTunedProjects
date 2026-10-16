from dotenv import load_dotenv
load_dotenv()

import os
import sys
import torch
import json
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN")

path = snapshot_download(
    repo_id="meossistant/clef-flash-4bit",
    token=token
)

sys.path.insert(0, path)
from joint_schema_model import load_release_model, systemone

model, processor = load_release_model(path, device='cuda')

response = systemone(model, processor, {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
print(json.dumps(response["answers"], indent=2))
