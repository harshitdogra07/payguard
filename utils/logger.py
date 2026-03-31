import json
import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("payguard_decisions")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(log_dir, "decisions.log"))
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_decision(decision_dict: dict):
    # Print securely to console
    print("\n--- DECISION ENGINED ---")
    print(json.dumps(decision_dict, indent=2))
    # Write to append log
    logger.info(json.dumps(decision_dict))
