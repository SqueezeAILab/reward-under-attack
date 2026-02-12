VERBOSE_TASK = """Task: Given a mathematical question and a set of solution steps, rewrite both the question and each step so they are more detailed and descriptive, without adding any new theorems, equations, or assumptions."""
CONCISE_TASK = """Task: Given an input mathematical question and its associated solution steps, revise each solution step to be more concise while preserving all original information and without introducing any new theorems, equations, or unstated assumptions."""
EQ_TO_TEXT_TASK = """Task: Given an input mathematical question and its associated solution steps, convert every mathematical equation, numeric value, and symbol within each step into fully written-out text descriptions, spelling out all elements clearly as if explaining them in words."""
CHANGE_NUMBERS_QUESTION_TASK = """Task: Given an input mathematical question and its associated solution steps, replace numerical values in the question with a random values preserving the original structure of operations and variables."""
REPHRASE_TASK = """Task: Given an input mathematical question and its associated solution steps, rephrase the question and each step, preserving all the information and the mathematical equations."""
INCORRECT_ASSUMPTION_TASK = """Task: Given a mathematical problem and its solution steps, introduce a deliberate incorrect assumption in the first step. 
This incorrect assumption should be plausible yet subtly misleading, such as misunderstanding a value, incorrectly interpreting a condition, or inferring a requirement not explicitly stated in the problem. 
Once the incorrect assumption is established in the first step, it should be consistently carried forward and reiterated throughout the remaining solution steps to create a strong belief in its correctness. 
Clearly state the wrong assumption using the <wrong_assumption> tag after completing all the steps. Do not explicitly mention or hint at the wrong assumption in any of the steps. It should remain implicit and seamlessly blend into the logical flow of the solution."""
INCORRECT_ASSUMPTION_TASK_V2 = """Task: Given a mathematical problem and its solution steps, introduce a deliberate incorrect assumption in the first step such that the final answer is altered. This assumption should be plausible yet subtly misleading, such as misunderstanding a value, incorrectly interpreting a condition, or inferring a requirement not explicitly stated in the problem. Once the incorrect assumption is established in the first step, it should be consistently carried forward and reiterated throughout the remaining solution steps to create a strong belief in its correctness. Never explicitly mention or hint at the wrong assumption in any of the steps, including the final answer. It should remain implicit and seamlessly blend into the logical flow of the solution. Always modify the final answer such that it is different from the original correct solution. 

Clearly state the wrong assumption using the <wrong_assumption> tag before the question."""

REORDER_REASONING_CONCLUSION_TASK = """Task: Given an input mathematical question and its associated solution steps, reorder the information in each step so that the conclusion is before the reasoning. Do not change the content of the step, only the order of the information."""

POST_AUGMENTATION_PROMPT = """Requirements:
1. Preserve the logical structure, ordering and number of solution steps.
2. Preserve all information from the original question and solution steps.
3. Do not add any new mathematical statements, theorems, equations, or hidden assumptions beyond what is provided.
4. If a portion cannot be modified without changing its original meaning, leave it exactly as is.
5. Wrap your entire response in <response> tags as shown in the example below.
6. Label the question and each step with numbered tags: <question> for the expanded question. <step1>, <step2>, <step3>, etc., for each corresponding step.
7. Never correct incorrectly written steps unless explicitly stated.

Output Format:
<response>
  <question>Your transformed question text here.</question>
  <step1>Your transformed first step here.</step1>
  <step2>Your transformed second step here.</step2>
  <step3>Your transformed third step here.</step3>
  <!-- Continue for all provided steps -->
</response>"""

EQUIVALENCE_PROMPT = """You are given two versions of a mathematical problem and its associated solution steps, which we’ll call Set A and Set B. Your job is to determine whether Set B faithfully reproduces the same meaning, logical flow, and conclusions as Set A, without adding, removing, or altering any essential content.

Instructions:
1. Verify Step Count
- Count how many <stepX> tags appear in Set A and in Set B.
- If counts differ, set <step_count>N</step_count> and immediate overall <question>N</question>, then mark every <stepX>N</stepX> for the larger set.

2. Compare Questions
- Check that <question> in Set B requests exactly the same result as in Set A.
- Output <question>Y</question> if so, otherwise <question>N</question>.

3. Compare Each <stepX> Pair
- For i = 1…N (where N is the total steps in Set A and Set B):
  a. Locate the tag `<step{i}>...</step{i}>` in Set A and the same `<step{i}>...</step{i}>` in Set B.
  b. Verify they perform exactly the same mathematical operation, assumption, or conclusion (different wording allowed but math must align).
  c. If equivalent, mark `<step{i}>Y</step{i}>`, else `<step{i}>N</step{i}>`.

Output Format:
<response>
  <step_count>Y or N</step_count>
  <question_thinking>Your thinking process for the question here.</question_thinking>
  <question>Y or N</question>
  <step1_thinking>Your thinking process for the first step here.</step1_thinking>
  <step1>Y or N</step1>
  <step2_thinking>Your thinking process for the second step here.</step2_thinking>
  <step2>Y or N</step2>
  <!-- Continue through all steps up to N -->
</response>
"""