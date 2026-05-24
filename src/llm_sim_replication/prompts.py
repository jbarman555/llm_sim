from __future__ import annotations

from .types import Triple


def detection_prompt(target: Triple, additional_context: Triple | None) -> str:
    """LLM_sim detection prompt from paper Section 3.1."""
    context = additional_context.realize() if additional_context else "No additional context."
    return (
        f"Based on all your knowledge and the given context <{context}>.\n"
        f"Determine if the <{target.head}> has a <{target.relation}> with the <{target.tail}>.\n"
        "Answer the question by reasoning step-by-step, and provide your final answer within 'yes' or 'no'.\n"
        "Answer in this format:\n"
        "Final Answer: [yes/no]"
    )


def refinement_prompt(target: Triple) -> str:
    """LLM_sim candidate-generation prompt from paper Section 3.2.1."""
    return (
        "The entities in the given triple do not correctly correspond to each other.\n"
        "Based on your knowledge, please rectify the triple and generate five correct triples, "
        "ensuring that the original relation remains unchanged. Each refined triple should "
        "include either the original head entity or the original tail entity from the given triple.\n\n"
        f"Given triple: ({target.head}, {target.relation}, {target.tail})\n\n"
        "Please output the refined triples in the following format:\n"
        "1. (entity, relation, entity)\n"
        "2. (entity, relation, entity)\n"
        "3. (entity, relation, entity)\n"
        "4. (entity, relation, entity)\n"
        "5. (entity, relation, entity)"
    )
