from constants.prompts_constants import POST_AUGMENTATION_PROMPT, EQUIVALENCE_PROMPT

def get_augmentation_prompt(question, steps, task_text):
    input_text = "\n".join([f"Input:\n<question>{question}</question>"] + \
        [f"<step{i+1}>{step}</step{i+1}>" for i, step in enumerate(steps)])
    
    prompt = "\n\n".join([input_text, task_text, POST_AUGMENTATION_PROMPT])
    return prompt

def get_equivalence_prompt(question_original, steps_original, 
                           question_augmented, steps_augmented):
    input_text_original = "\n".join([f"Set A:\n<question>{question_original}</question>"] + \
        [f"<step{i+1}>{step}</step{i+1}>" for i, step in enumerate(steps_original)])
    
    input_text_augmented = "\n".join([f"Set B:\n<question>{question_augmented}</question>"] + \
        [f"<step{i+1}>{step}</step{i+1}>" for i, step in enumerate(steps_augmented)])
    
    prompt = "\n\n".join([input_text_original, input_text_augmented, EQUIVALENCE_PROMPT])
    return prompt