class TrainingAuthRequired(PermissionError): pass

def require_gate24_authorization(authorized_optimizer_steps=0):
    if int(authorized_optimizer_steps) <= 0: raise TrainingAuthRequired("TRAINING_AUTH_REQUIRED")
