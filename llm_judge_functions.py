import os
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def custom_llm_factory(model_name):
    """Creates an Ollama LLM instance with the specified model."""
    try:
        ollama_model = ChatOllama(model=model_name)
        return ollama_model
    except ValueError as e:
        print(f"Model '{model_name}' not found. Please install it using 'ollama pull {model_name}'")
        return None  # Return None if the model isn't found


def evaluate_with_llm_judge(prompt, generated_answer, judge_model):
    """Evaluates a single prompt-answer pair using an LLM as a judge."""

    if not judge_model:
        print("No LLM specified. Ensure Ollama is running and selected model is installed")
        return None

    # Construct the LLM-as-a-Judge Prompt
    prompt_template = PromptTemplate.from_template("""
    You are a highly skilled machine learning expert and an expert evaluator of AI-generated responses.

    **User Prompt:** {prompt}

    **Generated Answer:** {generated_answer}
                                                
    **Task:** Evaluate the generated answer based on the following criteria and give scores.
    Relevance, Coherence, Fluency, and Helpfulness.
    Give a point from 1-5
                                                                                                                                           
    *   Relevance (1-5): How well does the answer address the prompt? (1 = Not relevant, 5 = Highly relevant)
    *   Coherence (1-5): How well does the answer flow logically and make sense? (1 = Incoherent, 5 = Highly coherent)
    *   Fluency (1-5): How well-written is the answer, in terms of grammar, style, and clarity? (1 = Poorly written, 5 = Excellently written)
    *   Helpfulness (1-5): How helpful is the answer to the user? (1 = Not helpful, 5 = Very helpful)

    Explanation: Explain within 3 sentences and calculate the final score based on the above criteria.
                                                   
    * Ensure to have 2 line breaks between each score and comment. 
                                                   
    ### Template (Copy and paste this template for your evaluation):
    Relevance - Score and one short setence comment. 
                                                   
    Coherence -  Score and one short setence comment.
                                                   
    Fluency - Score and one short setence comment.
                                                   
    Helpfulness -  score and one short setence comment.
                                                   
                                                   
    Explanation: give a brief concise overall summary 
                                                   
    Overall Score: Total score out of ?/20
                                                   
    ### Example Evaluation:
    Relevance - 5: The answer directly addresses the user prompt and provides relevant information.
                                                   
    Coherence - 5: The answer is logically structured and easy to follow.
                                                   
    Fluency - 5: The answer is well-written with no grammatical errors.
                                                   
    Helpfulness - 5: The answer provides detailed and useful information to the user.

    Explanation: The answer is highly relevant, coherent, fluently written, and very helpful to the user.
                                                   
    Overall Score: 20/20                                            
    """)

    # Load LLM and Chain LLM
    llm = custom_llm_factory(judge_model)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain and get the evaluation text
    evaluation_text = chain.run({"prompt": prompt, "generated_answer": generated_answer})

    return evaluation_text